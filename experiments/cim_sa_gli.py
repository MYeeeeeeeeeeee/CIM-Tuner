import math as m
from random import random, seed, getstate
from copy import deepcopy
from evacim import cim_acc_config as cfg

def func_cima(acc0, gli, dataflow):
    from evacim import sim
    m = sim.evaluate(acc0, gli, dataflow)
    return -m['throughput'], m['area'][7]*1e-6

def get_random_bool():
    dice = random()
    if dice<1/3: 
        return -1
    elif dice < 2/3: 
        return 0
    else:
        return 1

class SimulateAnnealing_CIMACC_on_gli:
    def __init__(self, func, acc0, gli, dataflow, iter=5, T0=100, Tf=1e-8, alpha=0.99, epoch=1, input_weight_switch_enable=True,input_weight_scheduling_enable = True, weight_tiling_enable = True,
                 fixed_hardware=False, proposal_budget=8, event_sink=None,
                 grid_domain=None, binding_domain=None):
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
        self.glis = [gli for i in range(self.iter)]
        self.dataflows = [dataflow for i in range(self.iter)]

        self.area_const = 5       
        self.ms_ratio = 0.08
        # 50 for ee_L2
        # 0.05 for throughput
        self.delta = {
            'macro_row': 1,
            'macro_col': 1,
            'scr_pow': 1,
            'is_size_pow': 1,
            'os_size_pow': 1
        }
        self.df_var = (input_weight_switch_enable, input_weight_scheduling_enable, weight_tiling_enable)
        self.history = {'f': [], 'T': [], 'c': []}
        self.fixed_hardware = fixed_hardware
        self.proposal_budget = proposal_budget
        self.event_sink = event_sink
        self.current = [None for _ in range(self.iter)]
        self.best = None
        self.proposals = 0
        self.failure = None
        self.reference_cycles = None
        self.grid_domain = None if grid_domain is None else tuple(tuple(g) for g in grid_domain)
        self.binding_domain = None if binding_domain is None else tuple(binding_domain)
        if self.grid_domain is not None:
            if (not fixed_hardware or not self.grid_domain or
                len(set(self.grid_domain)) != len(self.grid_domain) or
                any(len(g) != 2 or any(type(v) is not int for v in g) or
                    g not in ((1, 8), (2, 4), (4, 2), (8, 1)) for g in self.grid_domain)):
                raise ValueError("mapping 模式需要非空且不重複的固定八 Macro grid domain")
            if (acc0.macro_row, acc0.macro_col) not in self.grid_domain or not hasattr(acc0, 'binding'):
                raise ValueError("初始 mapping 不屬於 domain 或缺少部署 metadata")
        if self.binding_domain is not None:
            if (self.grid_domain is None or not self.binding_domain or
                len(set(self.binding_domain)) != len(self.binding_domain) or
                any(p not in ('identity', 'slot_major') for p in self.binding_domain) or
                acc0.binding_policy not in self.binding_domain):
                raise ValueError("binding domain 或初始 policy 不合法")
        if fixed_hardware:
            if (type(iter) is not int or iter <= 0 or type(epoch) is not int or epoch <= 0 or
                type(proposal_budget) is not int or proposal_budget <= 0 or
                not all(m.isfinite(v) for v in (T0, Tf, alpha)) or
                not 0 < Tf < T0 or not 0 < alpha < 1):
                raise ValueError("固定模式需要正預算與有限退火參數")
            if (len(gli) not in (2, 3) or len(gli) == 3 and gli[2] not in ('NR', 'R') or
                input_weight_switch_enable and len(gli) != 3):
                raise ValueError("固定模式開啟 switch 時需要明確 NR/R role")

    def generate_new_acc(self, acc0):
        if self.fixed_hardware:
            if self.grid_domain is not None:
                # 對稱重抽含 self-loop；每維一次抽樣，不進入 area 或重試迴圈。
                grid = self.grid_domain[int(random() * len(self.grid_domain))]
                acc = deepcopy(acc0)
                acc.macro_row, acc.macro_col = grid
                acc.AL = acc.cim.single_macro_al * acc.macro_col
                acc.PC = acc.cim.single_macro_pc * acc.macro_row
                if self.binding_domain is not None:
                    acc.binding_policy = self.binding_domain[int(random() * len(self.binding_domain))]
                    acc.binding = (list(range(8)) if acc.binding_policy == 'identity' else [0, 2, 4, 6, 1, 3, 5, 7])
                return acc
            return acc0
        from evacim import area_modeling
        while True:
            macro_row_new = acc0.macro_row + get_random_bool()*self.delta['macro_row']
            macro_col_new = acc0.macro_col + get_random_bool()*self.delta['macro_col']
            scr_new_pow = m.log2(acc0.SCR) + get_random_bool()*self.delta['scr_pow']
            is_size_new_pow = m.log2(acc0.is_size) + get_random_bool()*self.delta['is_size_pow']
            os_size_new_pow = m.log2(acc0.os_size) + get_random_bool()*self.delta['os_size_pow']
            
            
            if macro_row_new > 0 and macro_col_new > 0 and scr_new_pow > 0 and is_size_new_pow > 0 and os_size_new_pow > 0:
                acc_new = cfg.CIMACC(
                    cim = acc0.cim,
                    bus_width = acc0.bus_width,
                    macro_row = macro_row_new,
                    macro_col = macro_col_new,
                    scr = int(m.pow(2, scr_new_pow)),
                    is_size = int(m.pow(2, is_size_new_pow)),
                    os_size = int(m.pow(2, os_size_new_pow)),
                    )
                area = area_modeling.area_modeling(acc_new)
                if area[7]*1e-6 <= self.area_const:
                    break
        return acc_new

    def generate_new_gli(self, gli):
        if random()>0.5 and self.df_var[0]:
            if self.fixed_hardware and len(gli) == 3:
                return (gli[0], (gli[1][2], gli[1][1], gli[1][0]),
                        'R' if gli[2] == 'NR' else 'NR')
            return (gli[0],(gli[1][2], gli[1][1], gli[1][0]))
        else:
            return gli

    def generate_new_dataflow(self, dataflow):
        if self.df_var[1] and not self.df_var[2]:
            if random()>0.5: return 'is'+dataflow[2:]
            else: return 'ws'+dataflow[2:]
        elif self.df_var[1] and self.df_var[2]:
            dice = random()
            if dice>0 and dice <= 0.25: return 'isap'
            elif dice>0.25 and dice <= 0.5: return 'ispp'
            elif dice>0.5 and dice <= 0.75: return 'wsap'
            else: return 'wspp'
        else: return dataflow

    def Metrospolis(self, f, f_new):
        if f_new <= f:
            self.last_decision = {'probability': 1.0, 'random': None}
            return 1
        else:
            p = m.exp((f - f_new) * self.ms_ratio / self.T)
            draw = random()
            self.last_decision = {'probability': p, 'random': draw}
            if draw < p:
                return 1
            else:
                return 0

    def get_optimal(self):
        if self.fixed_hardware:
            valid = [(item['cycles'], i) for i, item in enumerate(self.current) if item]
            if not valid:
                return None, None, None
            score, idx = min(valid)
            return score, self.current[idx]['evaluation'], idx
        f_list = []
        area_list = []
        for i in range(self.iter):
            f, area = self.func(acc0 = self.accs[i], gli = self.glis[i], dataflow = self.dataflows[i])
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
        plt.savefig("./plot/sa_gli.png")

    def _emit(self, event, **fields):
        if self.event_sink:
            self.event_sink(deepcopy(dict(event=event, **fields)))

    def _mapping_fields(self, acc):
        if self.grid_domain is None:
            return {}
        return dict(mapping=dict(grid=[acc.macro_row, acc.macro_col], binding_policy=acc.binding_policy,
            binding=deepcopy(acc.binding), deployment_source=acc.deployment_source))

    def _evaluate_fixed(self, acc, gli, dataflow):
        try:
            score, record = self.func(acc0=acc, gli=gli, dataflow=dataflow)
            if not isinstance(record, dict) or record.get('status') not in ('passed', 'rejected', 'failed'):
                raise ValueError("callback 回傳狀態不完整")
            if record['status'] != 'passed':
                if score is not None or not record.get('reason'):
                    raise ValueError("拒絕／失敗不得回傳分數或省略原因")
                if record['status'] == 'failed':
                    self.failure = record['reason']
                self._emit('evaluation', dataflow=dataflow, evaluation=record)
                return None
            if type(score) is not int or score <= 0 or record.get('functional_pass') is not True:
                raise ValueError("成功評估必須包含正整數 cycles 與 golden 通過")
            if record.get('totalMeasuredCycles') != score:
                raise ValueError("callback 分數與實測 cycles 不一致")
            result = deepcopy(dict(acc=acc.backend_config, gli=gli, dataflow=dataflow,
                                   cycles=score, evaluation=record, **self._mapping_fields(acc)))
            if self.grid_domain is not None and record.get('mapping') != result['mapping']:
                raise ValueError("callback mapping 與原候選不符")
            if len(gli) == 3:
                if record.get('operand_role') != gli[2]:
                    raise ValueError("callback operand role 與原候選不符")
                result['operand_role'] = gli[2]
            if self.reference_cycles is None:
                self.reference_cycles = score
                self.ms_ratio = 1.0 / score
            if self.best is None or score < self.best['cycles']:
                self.best = deepcopy(result)
            self._emit('evaluation', result=result)
            return result
        except Exception as error:
            self.failure = str(error)
            self._emit('failure', reason=self.failure)
            return None

    def _fixed_result(self, stop_reason):
        reason = self.failure or (stop_reason if self.best else 'no_valid_solution')
        return dict(status='incomplete' if self.failure or self.best is None else 'complete',
                    reason=reason, best=None if self.failure else deepcopy(self.best),
                    best_observed=deepcopy(self.best), proposals=self.proposals,
                    reference_cycles=self.reference_cycles, history=deepcopy(self.history))

    def run(self):
        count = 0
        if self.fixed_hardware:
            for i in range(self.iter):
                self.current[i] = self._evaluate_fixed(self.accs[i], self.glis[i], self.dataflows[i])
                if self.failure:
                    return self._fixed_result('initial_failure')
        # annealing
        while self.epoch != 0:
            # iteration
            for i in range(self.iter):
                if self.fixed_hardware:
                    if self.proposals >= self.proposal_budget:
                        return self._fixed_result('proposal_budget')
                    before = deepcopy(self.current[i])
                    best_before = deepcopy(self.best)
                    rng_before = getstate()
                    f = before['cycles'] if before else None
                else:
                    f, _ = self.func(acc0 = self.accs[i], gli = self.glis[i], dataflow = self.dataflows[i])
                acc_new = self.generate_new_acc(self.accs[i])
                gli_new = self.generate_new_gli(self.glis[i])
                dataflow_new = self.generate_new_dataflow(self.dataflows[i])
                if self.fixed_hardware:
                    self.proposals += 1
                    self._emit('proposal', ordinal=self.proposals, chain=i, current=before,
                        acc=acc_new.backend_config, gli=gli_new, dataflow=dataflow_new,
                        **self._mapping_fields(acc_new),
                        methods=['generate_new_acc', 'generate_new_gli', 'generate_new_dataflow'],
                        rng_before=rng_before, rng_after=getstate())
                    new = self._evaluate_fixed(acc_new, gli_new, dataflow_new)
                    if self.failure:
                        return self._fixed_result('execution_failure')
                    accepted = False
                    self.last_decision = {'probability': None, 'random': None}
                    if new is not None:
                        accepted = f is None or bool(self.Metrospolis(f, new['cycles']))
                    if accepted:
                        self.current[i] = new
                    self._emit('decision', ordinal=self.proposals, chain=i, temperature=self.T,
                        reference_cycles=self.reference_cycles, before=before, proposal=new,
                        accepted=accepted, after=self.current[i], best_before=best_before,
                        best_after=self.best, **self.last_decision)
                else:
                    f_new, _ = self.func(acc0 = acc_new, gli = gli_new, dataflow = dataflow_new)
                    accepted = self.Metrospolis(f, f_new)
                if accepted:
                    self.accs[i] = acc_new
                    self.glis[i] = gli_new
                    self.dataflows[i] = dataflow_new
            # save to history
            ft, _, idx = self.get_optimal()
            self.history['f'].append(ft)
            self.history['T'].append(self.T)
            self.history['c'].append(count)
            if not self.fixed_hardware:
                print(ft, self.T,'\tidx:',idx, self.glis[idx], self.dataflows[idx], self.accs[idx].macro_row, self.accs[idx].macro_col, self.accs[idx].SCR, self.accs[idx].is_size, self.accs[idx].os_size)
            # cooling
            self.T = self.T * self.alpha
            count += 1
            if self.T < self.Tf:
                self.epoch -= 1
                self.T = self.T0 
        if self.fixed_hardware:
            return self._fixed_result('temperature')
        # get optimal solution
        f_best, area, idx = self.get_optimal()
        self.accs[idx].printf()
        print(f"best={f_best}, area={area}", count)

if __name__ == '__main__':
 
    cim = cfg.CIM("./cim_config/FPCIM@ISSCC23.cfg")
    gli = ('proj', [512,512,512])

    acc0 = cfg.CIMACC(
        cim,
        bus_width = 25.6, 
        macro_row = 3, 
        macro_col = 1, 
        scr = 16, 
        is_size = 64, 
        os_size = 64, 
        )

    seed(42)
    sa = SimulateAnnealing_CIMACC_on_gli(
        func_cima, 
        acc0, 
        gli, 
        "isap", 
        iter=5, 
        Tf=1, 
        alpha=0.99,
        epoch = 1, 
        input_weight_switch_enable = True,
        input_weight_scheduling_enable = True,
        weight_tiling_enable = True
    )

    sa.run()
    sa.plot()
    

