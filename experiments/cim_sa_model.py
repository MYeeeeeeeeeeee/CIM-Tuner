import math as m
from random import random, seed, getstate
from copy import deepcopy
from evacim import cim_acc_config as ccfg
from evacim import nn_config as ncfg

from evacim.sw_func import get_random_bool, func_cima, func_model, func_model_fix_dataflow


class SimulateAnnealing_CIMACC_on_model:
    def __init__(self, func, acc0, model, opt_target, area_const=5e6, iter=5, T0=100, Tf=1e-8, alpha=0.99, epoch=1,
                 fixed_hardware=False, grid_domain=None, proposal_budget=2, event_sink=None):
        """
        Annealing parameters
        :param iter: Number of internal cycles
        :param T0: Initial temperature
        :param Tf: Final temperature
        :param alpha: Cooling factor
        """
        self.func = func
        self.iter = iter
        self.alpha = alpha
        self.T0 = T0
        self.Tf = Tf
        self.T = T0
        self.epoch = epoch

        self.accs = [acc0 for i in range(self.iter)]
        self.model = model
        self.opt_target = opt_target

        self.area_const = area_const       
        self.ms_ratio = 50 if opt_target == 'ee_L2' else 0.05
        # 50 for ee_L2
        # 0.05 for throughput
        self.delta = {
            'macro_row': 1,
            'macro_col': 1,
            'scr_pow': 1,
            'is_size_pow': 1,
            'os_size_pow': 1
        }
        self.history = {'f': [], 'T': [], 'c': []}
        self.fixed_hardware = fixed_hardware
        if fixed_hardware:
            domain = ((2, 4), (4, 2)) if grid_domain is None else tuple(map(tuple, grid_domain))
            if (opt_target != 'backend_cycles' or not domain or len(set(domain)) != len(domain) or
                any(g not in ((2, 4), (4, 2)) for g in domain) or
                type(proposal_budget) is not int or proposal_budget < 1 or iter != 1 or
                not 0 < alpha < 1 or not 0 < Tf < T0 or type(epoch) is not int or epoch < 1):
                raise ValueError('限定 model 搜尋的候選域／退火設定不合法')
            self.grid_domain = domain
            self.proposal_budget, self.event_sink = proposal_budget, event_sink
            self.proposals = 0
            self.current, self.best, self.reference_cycles, self.failure = None, None, None, None
            self.accs = [deepcopy(acc0)]

    def generate_new_acc(self, acc0):
        if self.fixed_hardware:
            grid = self.grid_domain[min(int(random()*len(self.grid_domain)), len(self.grid_domain)-1)]
            return ccfg.CIMACC.from_backend_config(deepcopy(acc0.backend_config), grid=grid, binding_policy='identity')
        from evacim import area_modeling
        while True:
            macro_row_new = acc0.macro_row + get_random_bool()*self.delta['macro_row']
            macro_col_new = acc0.macro_col + get_random_bool()*self.delta['macro_col']
            scr_new_pow = m.log2(acc0.SCR) + get_random_bool()*self.delta['scr_pow']
            is_size_new_pow = m.log2(acc0.is_size) + get_random_bool()*self.delta['is_size_pow']
            os_size_new_pow = m.log2(acc0.os_size) + get_random_bool()*self.delta['os_size_pow']
            
            if macro_row_new > 0 and macro_col_new > 0 and scr_new_pow > 0 and is_size_new_pow > 0 and os_size_new_pow > 0:
                acc_new = ccfg.CIMACC(
                    cim = acc0.cim,
                    bus_width = acc0.bus_width,
                    macro_row = macro_row_new,
                    macro_col = macro_col_new,
                    scr = int(m.pow(2, scr_new_pow)),
                    is_size = int(m.pow(2, is_size_new_pow)),
                    os_size = int(m.pow(2, os_size_new_pow)),
                    )
                area = area_modeling.area_modeling(acc_new)
                if area[7] <= self.area_const:
                    break
        return acc_new

    def Metrospolis(self, f, f_new):
        self.last_decision = dict(probability=1.0, random=None)
        if f_new <= f:
            return 1
        else:
            p = m.exp((f - f_new) * self.ms_ratio / self.T)
            draw = random()
            self.last_decision = dict(probability=p, random=draw)
            if draw < p:
                return 1
            else:
                return 0

    def get_optimal(self):
        if self.fixed_hardware:
            return (self.best['cycles'], deepcopy(self.best['evaluation']), 0) if self.best else (None, None, None)
        f_list = []
        area_list = []
        for i in range(self.iter):
            f, _, area = self.func(acc0 = self.accs[i], model = self.model, opt_target = self.opt_target)
            f_list.append(f)
            area_list.append(area)
        f_best = min(f_list)
        idx = f_list.index(f_best)
        return - f_best, area_list[idx], idx

    def plot(self, xlim=None, ylim=None):
        import matplotlib.pyplot as plt
        plt.plot(self.history['c'], self.history['f'])
        plt.title('Simulate Annealing')
        # plt.xlabel('Temperature')
        plt.xlabel('Count')
        plt.ylabel('f value')
        if xlim:
            plt.xlim(xlim[0], xlim[-1])
        if ylim:
            plt.ylim(ylim[0], ylim[-1])
        # plt.gca().invert_xaxis()
        plt.show()
        plt.savefig("./plot/sa_model.png")

    def _emit(self, event, **fields):
        if self.event_sink:
            self.event_sink(deepcopy(dict(event=event, **fields)))

    def _evaluate_fixed(self, acc):
        try:
            score, choices, record = self.func(acc0=acc, model=self.model, opt_target=self.opt_target)
            if not isinstance(record, dict) or record.get('status') not in ('passed', 'rejected', 'failed'):
                raise ValueError('model callback狀態不完整')
            if record['status'] != 'passed':
                if score is not None or not record.get('reason'):
                    raise ValueError('model拒絕／失敗不得有分數或缺原因')
                if record['status'] == 'failed': self.failure = record['reason']
                self._emit('evaluation', evaluation=record)
                return None
            if (type(score) is not int or score <= 0 or record.get('functional_pass') is not True or
                record.get('model_cycles') != score or not choices):
                raise ValueError('model callback缺少正實測總分／完整choices')
            item = dict(cycles=score, grid=[acc.macro_row, acc.macro_col], binding=deepcopy(acc.binding),
                        choices=deepcopy(choices), evaluation=deepcopy(record))
            if record.get('grid') != item['grid'] or record.get('binding') != item['binding']:
                raise ValueError('model callback與proposal部署不符')
            if self.reference_cycles is None:
                self.reference_cycles = score
                self.ms_ratio = 1.0/score
            if self.best is None or score < self.best['cycles']: self.best = deepcopy(item)
            self._emit('evaluation', result=item)
            return item
        except Exception as error:
            self.failure = str(error) or type(error).__name__
            self._emit('failure', reason=self.failure)
            return None

    def _result(self, reason):
        return dict(status='incomplete' if self.failure or self.best is None else 'complete',
            reason=self.failure or (reason if self.best else 'no_valid_solution'),
            best=None if self.failure else deepcopy(self.best), best_observed=deepcopy(self.best),
            current=deepcopy(self.current), proposals=self.proposals, reference_cycles=self.reference_cycles,
            history=deepcopy(self.history))

    def run(self, write_log = False, logname = "res.log"):
        count = 0
        if self.fixed_hardware:
            self.current = self._evaluate_fixed(self.accs[0])
            if self.failure: return self._result('initial_failure')
        # annealing
        while self.epoch != 0:
            # iteration
            for i in range(self.iter):
                if self.fixed_hardware:
                    if self.proposals >= self.proposal_budget: return self._result('proposal_budget')
                    before, best_before = deepcopy(self.current), deepcopy(self.best)
                    rng_before = getstate()
                    acc_new = self.generate_new_acc(self.accs[i])
                    self.proposals += 1
                    self._emit('proposal', ordinal=self.proposals, method='generate_new_acc',
                        grid=[acc_new.macro_row, acc_new.macro_col], binding=acc_new.binding,
                        backend_config=acc_new.backend_config, rng_before=rng_before, rng_after=getstate(), current=before)
                    new = self._evaluate_fixed(acc_new)
                    if self.failure: return self._result('execution_failure')
                    self.last_decision = dict(probability=None, random=None)
                    accepted = new is not None and (before is None or bool(self.Metrospolis(before['cycles'], new['cycles'])))
                    if accepted: self.current, self.accs[i] = new, acc_new
                    self._emit('decision', ordinal=self.proposals, temperature=self.T, reference_cycles=self.reference_cycles,
                        before=before, proposal=new, accepted=accepted, after=self.current,
                        best_before=best_before, best_after=self.best, **self.last_decision)
                    continue
                f, _, _ = self.func(acc0 = self.accs[i], model = self.model, opt_target = self.opt_target)
                acc_new = self.generate_new_acc(self.accs[i])
                f_new, _, _ = self.func(acc0 = acc_new, model = self.model, opt_target = self.opt_target)
                if self.Metrospolis(f, f_new):
                    self.accs[i] = acc_new
            # save to history
            ft, _, idx = self.get_optimal()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            self.history['c'].append(count)
            if not self.fixed_hardware:
                print(ft, self.T,'\tidx:',idx, self.accs[idx].macro_row, self.accs[idx].macro_col, self.accs[idx].SCR, self.accs[idx].is_size, self.accs[idx].os_size, flush=True)
            # cooling
            self.T = self.T * self.alpha
            count += 1
            if self.T < self.Tf:
                self.epoch -= 1
                self.T = self.T0 
        # get optimal solution
        if self.fixed_hardware: return self._result('temperature')
        f_best, area, idx = self.get_optimal()
        self.accs[idx].printf()
        if write_log: 
            self.accs[idx].write_log(logname)
            with open(logname, 'a') as f:
                f.write(f"best={f_best}, area={area}, count={count}")
        print(f"best={f_best}, area={area}", count, flush=True)

if __name__ == '__main__':
    
    model = ncfg.NN("./nn_config/bert_base_sl64.csv") 
    cim = ccfg.CIM("./cim_config/FPCIM@ISSCC23.cfg")
    acc_trancim = ccfg.CIMACC(
        cim,
        bus_width = 25.6, 
        macro_row = 1, 
        macro_col = 3, 
        scr = 1, 
        is_size = 64, 
        os_size = 128, 
        ) 

    seed(43)
    sa = SimulateAnnealing_CIMACC_on_model(
        func_model_fix_dataflow, 
        acc_trancim, 
        model,
        opt_target = 'ee_L2', 
        iter=5, 
        Tf=1, 
        alpha=0.99,
        epoch = 1, 
    )

    sa.run(write_log=True, logname="./Result/test.log")
    sa.plot()
