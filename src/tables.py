import pandas

def denan(x):
    if type(x) is str:
        return x
    else:
        return ""

def latexfloat(x):
    return ("%.2f"%float(x)).replace("-", "$-$", 1)

def table2():
    f = open("figures/table2.tex", "w")
    f.write("\\begin{tabular}{lccc}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Model} & \\textbf{CA} & \\textbf{UA} & \\textbf{Bias} \\\\\n")
    f.write("\\midrule\n")
    f.write("\\textbf{Primary models} & & & \\\\\n")
    for model in primary_models:
        idx = list(cdbsu["model"]).index(model)
        f.write("\\quad %s & %s $\pm$ %s & %s $\pm$ %s & %s $\pm$ %s \\\\\n"%(
            cdbsu["model"][idx],
            latexfloat(cdbsu["confounded_test"][idx]),
            latexfloat(cdbsu["confounded_test_std"][idx]),
            latexfloat(cdbsu["unconfounded_test"][idx]),
            latexfloat(cdbsu["unconfounded_test_std"][idx]),
            latexfloat(cdbsu["bias"][idx]),
            latexfloat(cdbsu["bias_std"][idx])))
    f.write("\\textbf{Additional models} & & & \\\\\n")
    for model in additional_models:
        idx = list(cdbsu["model"]).index(model)
        f.write("\\quad %s & %s $\pm$ %s & %s $\pm$ %s & %s $\pm$ %s \\\\\n"%(
            cdbsu["model"][idx],
            latexfloat(cdbsu["confounded_test"][idx]),
            latexfloat(cdbsu["confounded_test_std"][idx]),
            latexfloat(cdbsu["unconfounded_test"][idx]),
            latexfloat(cdbsu["unconfounded_test_std"][idx]),
            latexfloat(cdbsu["bias"][idx]),
            latexfloat(cdbsu["bias_std"][idx])))
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.close()

def table3():
    f = open("figures/table3.tex", "w")
    f.write("\\begin{tabular}{lccc}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Model} & \\textbf{CA} & \\textbf{UA} & \\textbf{Bias} \\\\\n")
    f.write("\\midrule\n")
    f.write("\\textbf{Primary models} & & & \\\\\n")
    for model in primary_models:
        idx = list(pdbs["model"]).index(model)
        f.write("\\quad %s & %s $\pm$ %s & %s $\pm$ %s & %s $\pm$ %s \\\\\n"%(
            pdbs["model"][idx],
            latexfloat(pdbs["confounded_test"][idx]),
            latexfloat(pdbs["confounded_test_std"][idx]),
            latexfloat(pdbs["unconfounded_test"][idx]),
            latexfloat(pdbs["unconfounded_test_std"][idx]),
            latexfloat(pdbs["bias"][idx]),
            latexfloat(pdbs["bias_std"][idx])))
    f.write("\\textbf{Additional models} & & & \\\\\n")
    for model in additional_models:
        idx = list(pdbs["model"]).index(model)
        f.write("\\quad %s & %s $\pm$ %s & %s $\pm$ %s & %s $\pm$ %s \\\\\n"%(
            pdbs["model"][idx],
            latexfloat(pdbs["confounded_test"][idx]),
            latexfloat(pdbs["confounded_test_std"][idx]),
            latexfloat(pdbs["unconfounded_test"][idx]),
            latexfloat(pdbs["unconfounded_test_std"][idx]),
            latexfloat(pdbs["bias"][idx]),
            latexfloat(pdbs["bias_std"][idx])))
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.close()

def table4():
    f = open("figures/table4.tex", "w")
    f.write("\\begin{tabular}{lcc}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Model} & \\textbf{Bias estimate} & \\textbf{95\\% CI}\\\\\n")
    f.write("\\midrule\n")
    f.write("\\multicolumn{2}{l}{\\textbf{Primary models}}\\\\\n")
    for model in primary_models:
        idx = list(cdbsi["model"]).index(model)
        f.write("\\quad %s & %s\\rlap{\\textsuperscript{%s}} & [%s, %s]\\\\\n"%(
            cdbsi["model"][idx],
            latexfloat(cdbsi["estimate"][idx]),
            cdbsi["significance"][idx],
            latexfloat(cdbsi["lower_bound"][idx]),
            latexfloat(cdbsi["upper_bound"][idx])))
    f.write("\\multicolumn{2}{l}{\\textbf{Additional models}}\\\\\n")
    for model in additional_models:
        idx = list(cdbsi["model"]).index(model)
        f.write("\\quad %s & %s\\rlap{\\textsuperscript{%s}} & [%s, %s]\\\\\n"%(
            cdbsi["model"][idx],
            latexfloat(cdbsi["estimate"][idx]),
            cdbsi["significance"][idx],
            latexfloat(cdbsi["lower_bound"][idx]),
            latexfloat(cdbsi["upper_bound"][idx])))
    f.write("\\bottomrule\n")
    f.write("\\multicolumn{3}{p{0.7\\linewidth}}{\\small `***' indicates that the estimate is different from 1 at the $p < 0.001$ significance level.}\n")
    f.write("\\end{tabular}\n")
    f.close()

def table5():
    #\needswork: should index by name instead of [0], [1], [2], [3], and [4].
    f = open("figures/table5.tex", "w")
    f.write("\\begin{tabular}{lr@{\\hspace{3em}}c}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Parameter} &\n")
    f.write("\\multicolumn{1}{l}{\\textbf{Estimate}} &\n")
    f.write("\\textbf{95\\% CI} \\\\\n")
    f.write("\\midrule\n")
    f.write("\\textbf{Fixed effects} & & \\\\\n")
    f.write("\\quad $\\beta_{0}$ & %s\\rlap{\\textsuperscript{%s}} & [%s, %s] \\\\\n"%(
        latexfloat(fe["Estimate"][0]),
        fe["Sig"][0],
        latexfloat(ci["2.5 %"][3]),
        latexfloat(ci["97.5 %"][3])))
    f.write("\\quad $\\beta_{1}$ & %s\\rlap{\\textsuperscript{%s}} & [%s, %s] \\\\\n"%(
        latexfloat(fe["Estimate"][1]),
        fe["Sig"][1],
        latexfloat(ci["2.5 %"][4]),
        latexfloat(ci["97.5 %"][4])))
    f.write("\\textbf{Random effects} & & \\\\\n")
    f.write("\\quad $\\sigma^2_{m}$ & %s & [%s,  %s] \\\\\n"%(
        latexfloat(re["Var"][0]),
        latexfloat(ci["2.5 %"][0]),
        latexfloat(ci["97.5 %"][0])))
    f.write("\\quad $\\sigma^2_{s}$ & %s & [%s,  %s] \\\\\n"%(
        latexfloat(re["Var"][1]),
        latexfloat(ci["2.5 %"][1]),
        latexfloat(ci["97.5 %"][1])))
    f.write("\\quad $\\sigma^2$ & %s & [%s,  %s] \\\\\n"%(
        latexfloat(re["Var"][2]),
        latexfloat(ci["2.5 %"][2]),
        latexfloat(ci["97.5 %"][2])))
    f.write("\\bottomrule\n")
    f.write("\\multicolumn{3}{p{0.7\\linewidth}}{\\small `***' indicates that the estimate is different from 1 at the $p < 0.001$ significance level.}\n")
    f.write("\\end{tabular}\n")
    f.close()

def table6():
    f = open("figures/table6.tex", "w")
    f.write("\\begin{tabular}{lrc}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Parameter} & \\textbf{Estimate} & \\textbf{95\\% CI}\\\\\n")
    f.write("\\midrule\n")
    f.write("\\textbf{Fixed effects} & & \\\\\n")
    for parameter in ["Animal Body",
                      "Animal Face",
                      "Artificial Object",
                      "Human Body",
                      "Human Face",
                      "Natural Object"]:
        idx = list(cbe["category"]).index(parameter)
        f.write("\\quad %s & %s & [%s, %s] \\\\\n"%(
            cbe["category"][idx],
            latexfloat(cbe["Estimate"][idx]),
            latexfloat(cbe["2.5_ci"][idx]),
            latexfloat(cbe["97.5_ci"][idx])))
    f.write("\\textbf{Contrast} & & \\\\\n")
    for jdx, contrast in enumerate(["Animal Body - Animal Face",
                                    "Animal Body - Artificial Object",
                                    "Human Body - Animal Body",
                                    "Human Face - Animal Body",
                                    "Animal Body - Natural Object",
                                    "Animal Face - Artificial Object",
                                    "Human Body - Animal Face",
                                    "Human Face - Animal Face",
                                    "Animal Face - Natural Object",
                                    "Human Body - Artificial Object",
                                    "Human Face - Artificial Object",
                                    "Natural Object - Artificial Object",
                                    "Human Body - Human Face",
                                    "Human Body - Natural Object",
                                    "Human Face - Natural Object"]):
        idx = list(cbc["Contrast"]).index(contrast)
        f.write("\\quad %s & %s\\rlap{\\textsuperscript{%s}} & [%s, %s] \\\\\n"%(
            canonical_contrasts[jdx],
            latexfloat(cbc["Estimate"][idx]),
            denan(cbc["Sig"][idx]),
            latexfloat(cbc["2.5_ci"][idx]),
            latexfloat(cbc["97.5_ci"][idx])))
    f.write("\\textbf{Random effects} & &\\\\\n")
    #\needswork: should index by name instead of [0], [1], and [2]
    f.write("\\quad $\\sigma_{s}^2$ & %s & \\\\\n"%latexfloat(cbre["Var"][0]))
    f.write("\\quad $\\sigma_{m}^2$ & %s & \\\\\n"%latexfloat(cbre["Var"][1]))
    f.write("\\quad $\\sigma^2$ & %s & \\\\\n"%latexfloat(cbre["Var"][2]))
    f.write("\\bottomrule\n")
    f.write("\\multicolumn{3}{p{0.85\\linewidth}}{\\small `*', `**', and `***' indicate that the estimate is different from zero at the $p < 0.05$, $p < 0.01$, and $p < 0.001$ significance levels, respectively.}\n")
    f.write("\\end{tabular}\n")
    f.close()

def table7():
    f = open("figures/table7.tex", "w")
    f.write("\\begin{tabular}{l@{\\hspace{0em}}r@{\\hspace{3em}}c@{\\hspace{0em}}r@{\\hspace{2em}}c}\n")
    f.write("\\toprule\n")
    f.write("& \\multicolumn{2}{c}{\\textbf{Confounded}} & \\multicolumn{2}{c}{\\textbf{Unconfounded}}\\\\\n")
    f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n")
    f.write("\\textbf{Model} & \\multicolumn{1}{l}{\\textbf{Accuracy}} & \\textbf{95\\% CI} & \\multicolumn{1}{l}{\\textbf{Accuracy}} & \\textbf{95\\% CI}\\\\\n")
    f.write("\\midrule\n")
    f.write("\\textbf{Affected models}\\\\\n")
    for model in primary_models:
        idx = list(pdas["Unnamed: 0"]).index(model)
        f.write("\\quad %s & %s\\rlap{\\textsuperscript{%s}} & [%s, %s] & %s\\rlap{\\textsuperscript{%s}} & [%s, %s] \\\\\n"%(
            pdas["Unnamed: 0"][idx],
            latexfloat(pdas["estimate"][idx]),
            denan(pdas["significance"][idx]),
            latexfloat(pdas["lower_bound"][idx]),
            latexfloat(pdas["upper_bound"][idx]),
            latexfloat(pdas["estimate.1"][idx]),
            denan(pdas["significance.1"][idx]),
            latexfloat(pdas["lower_bound.1"][idx]),
            latexfloat(pdas["upper_bound.1"][idx])))
    f.write("\\textbf{Additional models} & & & & \\\\\n")
    for model in additional_models:
        idx = list(pdas["Unnamed: 0"]).index(model)
        f.write("\\quad %s & %s\\rlap{\\textsuperscript{%s}} & [%s, %s] & %s\\rlap{\\textsuperscript{%s}} & [%s, %s] \\\\\n"%(
            pdas["Unnamed: 0"][idx],
            latexfloat(pdas["estimate"][idx]),
            denan(pdas["significance"][idx]),
            latexfloat(pdas["lower_bound"][idx]),
            latexfloat(pdas["upper_bound"][idx]),
            latexfloat(pdas["estimate.1"][idx]),
            denan(pdas["significance.1"][idx]),
            latexfloat(pdas["lower_bound.1"][idx]),
            latexfloat(pdas["upper_bound.1"][idx])))
    f.write("\\bottomrule\n")
    f.write("\\multicolumn{5}{p{0.9\\linewidth}}{\\small `*', `**', and `***' indicate that the estimate is different from chance level (8.25\\%) at the $p < 0.05$, $p < 0.01$, and $p < 0.001$ significance levels, respectively.}\n")
    f.write("\\end{tabular}\n")
    f.close()

cbe = pandas.read_csv("outputs/paired_category_decoding/category_bias_estimates.csv")
cbc = pandas.read_csv("outputs/paired_category_decoding/category_bias_contrasts.csv")
cdbsi = pandas.read_csv("outputs/paired_category_decoding/category_decoding_bias_significance.csv")
cdbsu = pandas.read_csv("outputs/paired_category_decoding/category_decoding_bias_summary.csv")
ci = pandas.read_csv("outputs/paired_category_decoding/bias_accuracy_mixed_model_confidence_intervals.csv")
fe = pandas.read_csv("outputs/paired_category_decoding/bias_accuracy_mixed_model_fixed_effects.csv")
re = pandas.read_csv("outputs/paired_category_decoding/bias_accuracy_mixed_model_random_effects.csv")
pdas = pandas.read_csv("outputs/paired_pseudocategory_decoding/pseudocategory_decoding_accuracy_significance.csv")
pdbs = pandas.read_csv("outputs/paired_pseudocategory_decoding/pseudocategory_decoding_bias_summary.csv")
cbre = pandas.read_csv("outputs/paired_category_decoding/category_bias_random_effects.csv")

primary_models = ["LDA",
                  "ADCNN",
                  "AW1DCNN",
                  "EEGCT-Slim",
                  "EEGCT-Wide",
                  "RLSTM",
                  "STST",
                  "TSCNN"]
additional_models = ["LR",
                     "kNN",
                     "SVC",
                     "ShallowConvNet",
                     "DeepConvNet",
                     "EEGNet"]
canonical_contrasts = ["Animal Body---Animal Face",
                       "Animal Body---Artificial Object",
                       "Animal Body---Human Body",
                       "Animal Body---Human Face",
                       "Animal Body---Natural Object",
                       "Animal Face---Artificial Object",
                       "Animal Face---Human Body",
                       "Animal Face---Human Face",
                       "Animal Face---Natural Object",
                       "Artificial Object---Human Body",
                       "Artificial Object---Human Face",
                       "Artificial Object---Natural Object",
                       "Human Body---Human Face",
                       "Human Body---Natural Object",
                       "Human Face---Natural Object"]
table2()
table3()
table4()
table5()
table6()
table7()
