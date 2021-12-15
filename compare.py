from evaluate import *
import matplotlib.pyplot as plt
import numpy as np

def compare():
    eTE_lists, eRE_lists = [], []

    for method in ['datasets/VISIGRAPPTestVal0',
                   'datasets/VISIGRAPPTestVal1',
                   'datasets/VISIGRAPPTestVal2',
                   'datasets/VISIGRAPPTestVal3']:
        eTE_list, eRE_list, eGD_list, eTE_list_icp, eRE_list_icp, eGD_list_icp = evaluate(method)
        if method == 'datasets/VISIGRAPPTestVal0': #after icp
            eTE_list, eRE_list, eGD_list = eTE_list_icp, eRE_list_icp, eGD_list_icp

        eTE_lists.append(eTE_list)
        eRE_lists.append(eRE_list)

    eRE_maps, eTE_maps = [dict() for _ in range(len(eRE_lists))], [dict() for _ in range(len(eTE_lists))]
    count = len(eTE_lists[0])

    for method_num in range(4):
        for eRE in range(0, 301, 2):
            eRE = eRE / 100
            eRE_maps[method_num][eRE] = 0
            for sample_err in eRE_lists[method_num]:
                if sample_err < eRE:
                    eRE_maps[method_num][eRE] += 1
            eRE_maps[method_num][eRE] /= count

        for eTE in range(0, 501, 2):
            eTE = eTE / 10
            eTE_maps[method_num][eTE] = 0
            for sample_err in eTE_lists[method_num]:
                if sample_err < eTE:
                    eTE_maps[method_num][eTE] += 1
            eTE_maps[method_num][eTE] /= count

    print(eRE_maps[1])
    print(eTE_maps[1])

    plt.figure(figsize=(12, 3))
    ax1 = plt.subplot(121)
    ax1.set_xlabel('eRE')
    ax1.set_ylabel('portion of samples')
    #ax1.title.set_text('eRE')

    colors = ['b', 'g', 'r', 'k']
    markers = ['*', 'D', 'h', 'o']
    sizes = [10, 7, 7, 7]
    labels = ['After ICP refinement', 'ResNet34 1/2 full', 'ResNet34 1/2 w/o synth', 'Analytical method']
    for i in range(4):
        #plt.plot(eRE_maps[i].keys(), eRE_maps[i].values(), 'bo', eRE_maps[i].keys(), eRE_maps[i].values(), 'k')
        plt.xscale("log")
        ax1.plot(eRE_maps[i].keys(), eRE_maps[i].values(), colors[i], label=labels[i])
        #plt.plot(list(eRE_maps[i].keys())[::10], list(eRE_maps[i].values())[::10],
        #         colors[i] + markers[i], markersize=sizes[i], label=labels[i])
    plt.legend()


    ax2 = plt.subplot(122)
    ax2.set_xlabel('eTE')
    ax2.set_ylabel('portion of samples')
    #ax2.title.set_text('eTE')

    for i in range(4):
        plt.xscale("log")
        #ax2.set_xticks([1, 2, 5, 10])
        #ax2.set_xticklabels([1, 5, 10], fontsize=12)
        ax2.plot(eTE_maps[i].keys(), eTE_maps[i].values(), colors[i], label=labels[i])
        #plt.plot(list(eTE_maps[i].keys())[::10], list(eTE_maps[i].values())[::10],
        #         colors[i] + markers[i], markersize=sizes[i], label=labels[i])

    plt.legend()
    plt.savefig("Graph.pdf", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    compare()