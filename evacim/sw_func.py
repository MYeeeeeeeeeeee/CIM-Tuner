from random import random
import numpy as np
import argparse

from evacim import cim_acc_config as ccfg
from evacim import nn_config as ncfg

def func_model(acc0, model, opt_target, verbose = False, backend_evaluator=None):
    if backend_evaluator is not None:
        if opt_target != 'backend_cycles': raise ValueError('backend model只接受正cycles目標')
        records, choices, total = [], [], 0
        for dims, count in zip(model.gli_list_rdc, model.gli_cnt):
            gli = ('proj', tuple(int(x) for x in dims), 'NR')
            record, choice = block_decision(acc0, gli, opt_target, backend_evaluator=backend_evaluator)
            records.append(dict(dimensions=list(gli[1]), count=int(count), result=record, choice=choice))
            if record['status'] != 'passed':
                return None, choices, dict(status=record['status'], reason=record['reason'], shape_results=records)
            choices.append(choice)
            total += int(count)*record['totalMeasuredCycles']
        return total, choices, dict(status='passed', functional_pass=True, model_cycles=total, shape_results=records,
            grid=[acc0.macro_row, acc0.macro_col], binding=list(acc0.binding))
    num_ind_layer = len(model.gli_list_rdc)
    m_layer_rec = [[] for i in range(num_ind_layer)]
    df_rec = [[] for i in range(num_ind_layer)]

    total_energy = 0
    total_op = 0
    total_delay = 0

    if verbose:
        print("Operators List with different size: ")

    for i in range(num_ind_layer):
        dims = tuple(int(x) for x in model.gli_list_rdc[i])
        gli = ('proj', dims)
        m_layer_rec[i], df_rec[i] = block_decision(acc0, gli, opt_target)
        total_energy += m_layer_rec[i]['energy_L2']*model.gli_cnt[i]
        total_op += m_layer_rec[i]['op']*model.gli_cnt[i]
        total_delay += m_layer_rec[i]['latency']*model.gli_cnt[i]
        
        if verbose:
            print(i,": ",model.gli_list_rdc[i])
        # print(m_layer_rec[i]['ee_L2'])

    area = m_layer_rec[0]['area'][7]

    if opt_target == 'ee_L2':
        return - total_op/total_energy, df_rec, area
    elif opt_target == 'throughput':
        return - total_op/total_delay, df_rec, area
    elif opt_target == 'EDP':
        return total_delay*total_energy/1e12, df_rec, area
    else:
        return - total_op/total_energy, df_rec, area


def func_model_fix_dataflow(acc0, model, opt_target):
    num_ind_layer = len(model.gli_list_rdc)
    m_layer_rec = [[] for i in range(num_ind_layer)]
    df_rec = [[] for i in range(num_ind_layer)]

    total_energy = 0
    total_op = 0
    total_delay = 0

    for i in range(num_ind_layer):
        dims = tuple(int(x) for x in model.gli_list_rdc[i])
        gli = ('proj', dims)
        m_layer_rec[i], df_rec[i] = block_decision(acc0, gli, opt_target, fix = True)
        total_energy += m_layer_rec[i]['energy_L2']*model.gli_cnt[i]
        total_op += m_layer_rec[i]['op']*model.gli_cnt[i]
        total_delay += m_layer_rec[i]['latency']*model.gli_cnt[i]

    area = m_layer_rec[0]['area'][7]

    if opt_target == 'ee_L2':
        return - total_op/total_energy, df_rec, area
    elif opt_target == 'throughput':
        return - total_op/total_delay, df_rec, area
    elif opt_target == 'EDP':
        return total_delay*total_energy/1e12, df_rec, area
    else:
        return - total_op/total_energy, df_rec, area


def block_decision(acc0, gli, opt_target='ee_L2', fix=False, backend_evaluator=None):
    if backend_evaluator is not None:
        if opt_target != 'backend_cycles' or fix: raise ValueError('backend block設定不合法')
        best, best_choice, attempts = None, None, []
        for role in ('NR', 'R'):
            view = gli[1] if role == 'NR' else (gli[1][2], gli[1][1], gli[1][0])
            for df in ('isap', 'ispp', 'wsap', 'wspp'):
                score, result = backend_evaluator(acc0, ('proj', view, role), df)
                attempts.append(result)
                if result.get('status') == 'failed': return dict(result, attempts=attempts), None
                if result.get('status') == 'rejected':
                    if score is not None: raise ValueError('非法block有分數')
                    continue
                if (result.get('status') != 'passed' or type(score) is not int or score <= 0 or
                    result.get('functional_pass') is not True or result.get('totalMeasuredCycles') != score):
                    raise ValueError('backend block成功契約不符')
                if best is None or score < best['totalMeasuredCycles']:
                    best, best_choice = result, dict(role=role, dataflow=df, evaluation_id=result['evaluation_id'])
        if best is None: return dict(status='rejected', reason='必要shape沒有合法mapping', attempts=attempts), None
        return dict(best, attempts=attempts), best_choice
    from evacim import sim
    if fix == True:
        m_rec = [[] for i in range(4)]
        dataflow = ['isap','wsap','r_isap','r_wsap']
        m_rec[0] = sim.evaluate(acc0, gli, dataflow[0])
        m_rec[1] = sim.evaluate(acc0, gli, dataflow[1])
        rv_gli = (gli[0],(gli[1][2],gli[1][1],gli[1][0]))
        m_rec[2] = sim.evaluate(acc0, rv_gli, dataflow[0])
        m_rec[3] = sim.evaluate(acc0, rv_gli, dataflow[1])
        
        cmpme = np.zeros(4)
        for i in range(4):
            cmpme[i] = m_rec[i][opt_target]
        idx = np.argmin(cmpme)
    
        return m_rec[idx], dataflow[idx]
    
    else:
        m_rec = [[] for i in range(8)]
        dataflow = ['isap','ispp','wsap','wspp','r_isap','r_ispp','r_wsap','r_wspp']
        m_rec[0] = sim.evaluate(acc0, gli, 'isap')
        m_rec[1] = sim.evaluate(acc0, gli, 'ispp')
        m_rec[2] = sim.evaluate(acc0, gli, 'wsap')
        m_rec[3] = sim.evaluate(acc0, gli, 'wspp')

        rv_gli = (gli[0],(gli[1][2],gli[1][1],gli[1][0]))
        m_rec[4] = sim.evaluate(acc0, rv_gli, 'isap')
        m_rec[5] = sim.evaluate(acc0, rv_gli, 'ispp')
        m_rec[6] = sim.evaluate(acc0, rv_gli, 'wsap')
        m_rec[7] = sim.evaluate(acc0, rv_gli, 'wspp')
    
        cmpme = np.zeros(8)
        for i in range(8):
            cmpme[i] = m_rec[i][opt_target]
        idx = np.argmax(cmpme)
    
        return m_rec[idx], dataflow[idx]


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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cim", help="Choose the CIM cfg file")
    parser.add_argument("-p", "--paras", nargs='+', type=float, help="Set the parameters of Accelerator: BW, MR, MC, SCR, IS_SIZE, OS_SIZE")
    parser.add_argument("-m", "--model", help="Choose the model cfg file")
    parser.add_argument("-t", "--target", help="Choose the optimal target for different layers: ee for energy efficiency and th for throughput. ")

    args = parser.parse_args()

    cim = ccfg.CIM(args.cim)
    acc0 = ccfg.CIMACC(
       cim = cim,
       bus_width = args.paras[0], 
       macro_row = args.paras[1], 
       macro_col = args.paras[2], 
       scr = args.paras[3], 
       is_size = args.paras[4], 
       os_size = args.paras[5], 
    )

    model = ncfg.NN(args.model) 
    target = 'ee_L2' if args.target == 'ee' else 'throughput'

    print("\n######## Model Evaluation Results #######")
    m, df, area = func_model(acc0, model, 'throughput', verbose=True) 
    print("\nTargeting ", args.target, ", corresponding optimal mappings are:")
    # Notation configuration
    # NR_IP_AF => isap
    # NR_IP_PF => ispp
    # NR_WP_AF => wsap
    # NR_WP_PF => wspp
    # R_IP_AF => r_isap
    # R_IP_PF => r_ispp
    # R_WP_AF => r_wsap
    # R_WP_PF => r_wspp
    for i in range(len(df)):
        if df[i] == 'isap':
            print(i,": NR_IP_AF")
        elif df[i] == 'ispp':
            print(i,": NR_IP_PF")
        elif df[i] == 'wsap':
            print(i,": NR_WP_AF")
        elif df[i] == 'wspp':
            print(i,": NR_WP_PF")
        elif df[i] == 'r_isap':
            print(i,": R_IP_AF")
        elif df[i] == 'r_ispp':
            print(i,": R_IP_PF")
        elif df[i] == 'r_wsap':
            print(i,": R_WP_AF")
        elif df[i] == 'r_wspp':
            print(i,": R_WP_PF")
        else:
            pass
    Units = "TOPS/mm^2" if target == 'ee_L2' else "GOPS"
    print("\nTarget ("+Units+")",target,": ", -1*m)
    print("Area (mm^2):", area)
