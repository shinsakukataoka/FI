import numpy as np

ber_values = [
    1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9, 7e-9, 8e-9, 9e-9,
    1e-8, 2e-8, 3e-8, 4e-8, 5e-8, 6e-8, 7e-8, 8e-8, 9e-8,
    1e-7, 2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7,
    1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6,
    1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5,
    1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
    1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
    1e-2
]

# Interpolate to generate approximately 1000 values
interp_values = np.interp(np.linspace(0, len(ber_values) - 1, 1000), range(len(ber_values)), ber_values)

def divide_list(lst, num_parts):
    n = len(lst)
    division_size = n // num_parts
    divisions = [lst[i * division_size: (i + 1) * division_size] for i in range(num_parts)]
    return divisions

total_list = divide_list(interp_values,10)

list_1 = [1.00000000e-09, 1.07207207e-09, 1.14414414e-09, 1.21621622e-09,
       1.28828829e-09, 1.36036036e-09, 1.43243243e-09, 1.50450450e-09,
       1.57657658e-09, 1.64864865e-09, 1.72072072e-09, 1.79279279e-09,
       1.86486486e-09, 1.93693694e-09, 2.00900901e-09, 2.08108108e-09,
       2.15315315e-09, 2.22522523e-09, 2.29729730e-09, 2.36936937e-09,
       2.44144144e-09, 2.51351351e-09, 2.58558559e-09, 2.65765766e-09,
       2.72972973e-09, 2.80180180e-09, 2.87387387e-09, 2.94594595e-09,
       3.01801802e-09, 3.09009009e-09, 3.16216216e-09, 3.23423423e-09,
       3.30630631e-09, 3.37837838e-09, 3.45045045e-09, 3.52252252e-09,
       3.59459459e-09, 3.66666667e-09, 3.73873874e-09, 3.81081081e-09,
       3.88288288e-09, 3.95495495e-09, 4.02702703e-09, 4.09909910e-09,
       4.17117117e-09, 4.24324324e-09, 4.31531532e-09, 4.38738739e-09,
       4.45945946e-09, 4.53153153e-09, 4.60360360e-09, 4.67567568e-09,
       4.74774775e-09, 4.81981982e-09, 4.89189189e-09, 4.96396396e-09,
       5.03603604e-09, 5.10810811e-09, 5.18018018e-09, 5.25225225e-09,
       5.32432432e-09, 5.39639640e-09, 5.46846847e-09, 5.54054054e-09,
       5.61261261e-09, 5.68468468e-09, 5.75675676e-09, 5.82882883e-09,
       5.90090090e-09, 5.97297297e-09, 6.04504505e-09, 6.11711712e-09,
       6.18918919e-09, 6.26126126e-09, 6.33333333e-09, 6.40540541e-09,
       6.47747748e-09, 6.54954955e-09, 6.62162162e-09, 6.69369369e-09,
       6.76576577e-09, 6.83783784e-09, 6.90990991e-09, 6.98198198e-09,
       7.05405405e-09, 7.12612613e-09, 7.19819820e-09, 7.27027027e-09,
       7.34234234e-09, 7.41441441e-09, 7.48648649e-09, 7.55855856e-09,
       7.63063063e-09, 7.70270270e-09, 7.77477477e-09, 7.84684685e-09,
       7.91891892e-09, 7.99099099e-09, 8.06306306e-09, 8.13513514e-09]

list_2 = [8.20720721e-09, 8.27927928e-09, 8.35135135e-09, 8.42342342e-09,
       8.49549550e-09, 8.56756757e-09, 8.63963964e-09, 8.71171171e-09,
       8.78378378e-09, 8.85585586e-09, 8.92792793e-09, 9.00000000e-09,
       9.07207207e-09, 9.14414414e-09, 9.21621622e-09, 9.28828829e-09,
       9.36036036e-09, 9.43243243e-09, 9.50450450e-09, 9.57657658e-09,
       9.64864865e-09, 9.72072072e-09, 9.79279279e-09, 9.86486486e-09,
       9.93693694e-09, 1.00900901e-08, 1.08108108e-08, 1.15315315e-08,
       1.22522523e-08, 1.29729730e-08, 1.36936937e-08, 1.44144144e-08,
       1.51351351e-08, 1.58558559e-08, 1.65765766e-08, 1.72972973e-08,
       1.80180180e-08, 1.87387387e-08, 1.94594595e-08, 2.01801802e-08,
       2.09009009e-08, 2.16216216e-08, 2.23423423e-08, 2.30630631e-08,
       2.37837838e-08, 2.45045045e-08, 2.52252252e-08, 2.59459459e-08,
       2.66666667e-08, 2.73873874e-08, 2.81081081e-08, 2.88288288e-08,
       2.95495495e-08, 3.02702703e-08, 3.09909910e-08, 3.17117117e-08,
       3.24324324e-08, 3.31531532e-08, 3.38738739e-08, 3.45945946e-08,
       3.53153153e-08, 3.60360360e-08, 3.67567568e-08, 3.74774775e-08,
       3.81981982e-08, 3.89189189e-08, 3.96396396e-08, 4.03603604e-08,
       4.10810811e-08, 4.18018018e-08, 4.25225225e-08, 4.32432432e-08,
       4.39639640e-08, 4.46846847e-08, 4.54054054e-08, 4.61261261e-08,
       4.68468468e-08, 4.75675676e-08, 4.82882883e-08, 4.90090090e-08,
       4.97297297e-08, 5.04504505e-08, 5.11711712e-08, 5.18918919e-08,
       5.26126126e-08, 5.33333333e-08, 5.40540541e-08, 5.47747748e-08,
       5.54954955e-08, 5.62162162e-08, 5.69369369e-08, 5.76576577e-08,
       5.83783784e-08, 5.90990991e-08, 5.98198198e-08, 6.05405405e-08,
       6.12612613e-08, 6.19819820e-08, 6.27027027e-08, 6.34234234e-08]

list_3 = [6.41441441e-08, 6.48648649e-08, 6.55855856e-08, 6.63063063e-08,
       6.70270270e-08, 6.77477477e-08, 6.84684685e-08, 6.91891892e-08,
       6.99099099e-08, 7.06306306e-08, 7.13513514e-08, 7.20720721e-08,
       7.27927928e-08, 7.35135135e-08, 7.42342342e-08, 7.49549550e-08,
       7.56756757e-08, 7.63963964e-08, 7.71171171e-08, 7.78378378e-08,
       7.85585586e-08, 7.92792793e-08, 8.00000000e-08, 8.07207207e-08,
       8.14414414e-08, 8.21621622e-08, 8.28828829e-08, 8.36036036e-08,
       8.43243243e-08, 8.50450450e-08, 8.57657658e-08, 8.64864865e-08,
       8.72072072e-08, 8.79279279e-08, 8.86486486e-08, 8.93693694e-08,
       9.00900901e-08, 9.08108108e-08, 9.15315315e-08, 9.22522523e-08,
       9.29729730e-08, 9.36936937e-08, 9.44144144e-08, 9.51351351e-08,
       9.58558559e-08, 9.65765766e-08, 9.72972973e-08, 9.80180180e-08,
       9.87387387e-08, 9.94594595e-08, 1.01801802e-07, 1.09009009e-07,
       1.16216216e-07, 1.23423423e-07, 1.30630631e-07, 1.37837838e-07,
       1.45045045e-07, 1.52252252e-07, 1.59459459e-07, 1.66666667e-07,
       1.73873874e-07, 1.81081081e-07, 1.88288288e-07, 1.95495495e-07,
       2.02702703e-07, 2.09909910e-07, 2.17117117e-07, 2.24324324e-07,
       2.31531532e-07, 2.38738739e-07, 2.45945946e-07, 2.53153153e-07,
       2.60360360e-07, 2.67567568e-07, 2.74774775e-07, 2.81981982e-07,
       2.89189189e-07, 2.96396396e-07, 3.03603604e-07, 3.10810811e-07,
       3.18018018e-07, 3.25225225e-07, 3.32432432e-07, 3.39639640e-07,
       3.46846847e-07, 3.54054054e-07, 3.61261261e-07, 3.68468468e-07,
       3.75675676e-07, 3.82882883e-07, 3.90090090e-07, 3.97297297e-07,
       4.04504505e-07, 4.11711712e-07, 4.18918919e-07, 4.26126126e-07,
       4.33333333e-07, 4.40540541e-07, 4.47747748e-07, 4.54954955e-07]

list_4 = [4.62162162e-07, 4.69369369e-07, 4.76576577e-07, 4.83783784e-07,
       4.90990991e-07, 4.98198198e-07, 5.05405405e-07, 5.12612613e-07,
       5.19819820e-07, 5.27027027e-07, 5.34234234e-07, 5.41441441e-07,
       5.48648649e-07, 5.55855856e-07, 5.63063063e-07, 5.70270270e-07,
       5.77477477e-07, 5.84684685e-07, 5.91891892e-07, 5.99099099e-07,
       6.06306306e-07, 6.13513514e-07, 6.20720721e-07, 6.27927928e-07,
       6.35135135e-07, 6.42342342e-07, 6.49549550e-07, 6.56756757e-07,
       6.63963964e-07, 6.71171171e-07, 6.78378378e-07, 6.85585586e-07,
       6.92792793e-07, 7.00000000e-07, 7.07207207e-07, 7.14414414e-07,
       7.21621622e-07, 7.28828829e-07, 7.36036036e-07, 7.43243243e-07,
       7.50450450e-07, 7.57657658e-07, 7.64864865e-07, 7.72072072e-07,
       7.79279279e-07, 7.86486486e-07, 7.93693694e-07, 8.00900901e-07,
       8.08108108e-07, 8.15315315e-07, 8.22522523e-07, 8.29729730e-07,
       8.36936937e-07, 8.44144144e-07, 8.51351351e-07, 8.58558559e-07,
       8.65765766e-07, 8.72972973e-07, 8.80180180e-07, 8.87387387e-07,
       8.94594595e-07, 9.01801802e-07, 9.09009009e-07, 9.16216216e-07,
       9.23423423e-07, 9.30630631e-07, 9.37837838e-07, 9.45045045e-07,
       9.52252252e-07, 9.59459459e-07, 9.66666667e-07, 9.73873874e-07,
       9.81081081e-07, 9.88288288e-07, 9.95495495e-07, 1.02702703e-06,
       1.09909910e-06, 1.17117117e-06, 1.24324324e-06, 1.31531532e-06,
       1.38738739e-06, 1.45945946e-06, 1.53153153e-06, 1.60360360e-06,
       1.67567568e-06, 1.74774775e-06, 1.81981982e-06, 1.89189189e-06,
       1.96396396e-06, 2.03603604e-06, 2.10810811e-06, 2.18018018e-06,
       2.25225225e-06, 2.32432432e-06, 2.39639640e-06, 2.46846847e-06,
       2.54054054e-06, 2.61261261e-06, 2.68468468e-06, 2.75675676e-06]

list_5 = [2.82882883e-06, 2.90090090e-06, 2.97297297e-06, 3.04504505e-06,
       3.11711712e-06, 3.18918919e-06, 3.26126126e-06, 3.33333333e-06,
       3.40540541e-06, 3.47747748e-06, 3.54954955e-06, 3.62162162e-06,
       3.69369369e-06, 3.76576577e-06, 3.83783784e-06, 3.90990991e-06,
       3.98198198e-06, 4.05405405e-06, 4.12612613e-06, 4.19819820e-06,
       4.27027027e-06, 4.34234234e-06, 4.41441441e-06, 4.48648649e-06,
       4.55855856e-06, 4.63063063e-06, 4.70270270e-06, 4.77477477e-06,
       4.84684685e-06, 4.91891892e-06, 4.99099099e-06, 5.06306306e-06,
       5.13513514e-06, 5.20720721e-06, 5.27927928e-06, 5.35135135e-06,
       5.42342342e-06, 5.49549550e-06, 5.56756757e-06, 5.63963964e-06,
       5.71171171e-06, 5.78378378e-06, 5.85585586e-06, 5.92792793e-06,
       6.00000000e-06, 6.07207207e-06, 6.14414414e-06, 6.21621622e-06,
       6.28828829e-06, 6.36036036e-06, 6.43243243e-06, 6.50450450e-06,
       6.57657658e-06, 6.64864865e-06, 6.72072072e-06, 6.79279279e-06,
       6.86486486e-06, 6.93693694e-06, 7.00900901e-06, 7.08108108e-06,
       7.15315315e-06, 7.22522523e-06, 7.29729730e-06, 7.36936937e-06,
       7.44144144e-06, 7.51351351e-06, 7.58558559e-06, 7.65765766e-06,
       7.72972973e-06, 7.80180180e-06, 7.87387387e-06, 7.94594595e-06,
       8.01801802e-06, 8.09009009e-06, 8.16216216e-06, 8.23423423e-06,
       8.30630631e-06, 8.37837838e-06, 8.45045045e-06, 8.52252252e-06,
       8.59459459e-06, 8.66666667e-06, 8.73873874e-06, 8.81081081e-06,
       8.88288288e-06, 8.95495495e-06, 9.02702703e-06, 9.09909910e-06,
       9.17117117e-06, 9.24324324e-06, 9.31531532e-06, 9.38738739e-06,
       9.45945946e-06, 9.53153153e-06, 9.60360360e-06, 9.67567568e-06,
       9.74774775e-06, 9.81981982e-06, 9.89189189e-06, 9.96396396e-06]

list_6 = [1.03603604e-05, 1.10810811e-05, 1.18018018e-05, 1.25225225e-05,
       1.32432432e-05, 1.39639640e-05, 1.46846847e-05, 1.54054054e-05,
       1.61261261e-05, 1.68468468e-05, 1.75675676e-05, 1.82882883e-05,
       1.90090090e-05, 1.97297297e-05, 2.04504505e-05, 2.11711712e-05,
       2.18918919e-05, 2.26126126e-05, 2.33333333e-05, 2.40540541e-05,
       2.47747748e-05, 2.54954955e-05, 2.62162162e-05, 2.69369369e-05,
       2.76576577e-05, 2.83783784e-05, 2.90990991e-05, 2.98198198e-05,
       3.05405405e-05, 3.12612613e-05, 3.19819820e-05, 3.27027027e-05,
       3.34234234e-05, 3.41441441e-05, 3.48648649e-05, 3.55855856e-05,
       3.63063063e-05, 3.70270270e-05, 3.77477477e-05, 3.84684685e-05,
       3.91891892e-05, 3.99099099e-05, 4.06306306e-05, 4.13513514e-05,
       4.20720721e-05, 4.27927928e-05, 4.35135135e-05, 4.42342342e-05,
       4.49549550e-05, 4.56756757e-05, 4.63963964e-05, 4.71171171e-05,
       4.78378378e-05, 4.85585586e-05, 4.92792793e-05, 5.00000000e-05,
       5.07207207e-05, 5.14414414e-05, 5.21621622e-05, 5.28828829e-05,
       5.36036036e-05, 5.43243243e-05, 5.50450450e-05, 5.57657658e-05,
       5.64864865e-05, 5.72072072e-05, 5.79279279e-05, 5.86486486e-05,
       5.93693694e-05, 6.00900901e-05, 6.08108108e-05, 6.15315315e-05,
       6.22522523e-05, 6.29729730e-05, 6.36936937e-05, 6.44144144e-05,
       6.51351351e-05, 6.58558559e-05, 6.65765766e-05, 6.72972973e-05,
       6.80180180e-05, 6.87387387e-05, 6.94594595e-05, 7.01801802e-05,
       7.09009009e-05, 7.16216216e-05, 7.23423423e-05, 7.30630631e-05,
       7.37837838e-05, 7.45045045e-05, 7.52252252e-05, 7.59459459e-05,
       7.66666667e-05, 7.73873874e-05, 7.81081081e-05, 7.88288288e-05,
       7.95495495e-05, 8.02702703e-05, 8.09909910e-05, 8.17117117e-05]

list_7 = [8.24324324e-05, 8.31531532e-05, 8.38738739e-05, 8.45945946e-05,
       8.53153153e-05, 8.60360360e-05, 8.67567568e-05, 8.74774775e-05,
       8.81981982e-05, 8.89189189e-05, 8.96396396e-05, 9.03603604e-05,
       9.10810811e-05, 9.18018018e-05, 9.25225225e-05, 9.32432432e-05,
       9.39639640e-05, 9.46846847e-05, 9.54054054e-05, 9.61261261e-05,
       9.68468468e-05, 9.75675676e-05, 9.82882883e-05, 9.90090090e-05,
       9.97297297e-05, 1.04504505e-04, 1.11711712e-04, 1.18918919e-04,
       1.26126126e-04, 1.33333333e-04, 1.40540541e-04, 1.47747748e-04,
       1.54954955e-04, 1.62162162e-04, 1.69369369e-04, 1.76576577e-04,
       1.83783784e-04, 1.90990991e-04, 1.98198198e-04, 2.05405405e-04,
       2.12612613e-04, 2.19819820e-04, 2.27027027e-04, 2.34234234e-04,
       2.41441441e-04, 2.48648649e-04, 2.55855856e-04, 2.63063063e-04,
       2.70270270e-04, 2.77477477e-04, 2.84684685e-04, 2.91891892e-04,
       2.99099099e-04, 3.06306306e-04, 3.13513514e-04, 3.20720721e-04,
       3.27927928e-04, 3.35135135e-04, 3.42342342e-04, 3.49549550e-04,
       3.56756757e-04, 3.63963964e-04, 3.71171171e-04, 3.78378378e-04,
       3.85585586e-04, 3.92792793e-04, 4.00000000e-04, 4.07207207e-04,
       4.14414414e-04, 4.21621622e-04, 4.28828829e-04, 4.36036036e-04,
       4.43243243e-04, 4.50450450e-04, 4.57657658e-04, 4.64864865e-04,
       4.72072072e-04, 4.79279279e-04, 4.86486486e-04, 4.93693694e-04,
       5.00900901e-04, 5.08108108e-04, 5.15315315e-04, 5.22522523e-04,
       5.29729730e-04, 5.36936937e-04, 5.44144144e-04, 5.51351351e-04,
       5.58558559e-04, 5.65765766e-04, 5.72972973e-04, 5.80180180e-04,
       5.87387387e-04, 5.94594595e-04, 6.01801802e-04, 6.09009009e-04,
       6.16216216e-04, 6.23423423e-04, 6.30630631e-04, 6.37837838e-04]

list_8 = [0.00064505, 0.00065225, 0.00065946, 0.00066667, 0.00067387,
       0.00068108, 0.00068829, 0.0006955 , 0.0007027 , 0.00070991,
       0.00071712, 0.00072432, 0.00073153, 0.00073874, 0.00074595,
       0.00075315, 0.00076036, 0.00076757, 0.00077477, 0.00078198,
       0.00078919, 0.0007964 , 0.0008036 , 0.00081081, 0.00081802,
       0.00082523, 0.00083243, 0.00083964, 0.00084685, 0.00085405,
       0.00086126, 0.00086847, 0.00087568, 0.00088288, 0.00089009,
       0.0008973 , 0.0009045 , 0.00091171, 0.00091892, 0.00092613,
       0.00093333, 0.00094054, 0.00094775, 0.00095495, 0.00096216,
       0.00096937, 0.00097658, 0.00098378, 0.00099099, 0.0009982 ,
       0.00105405, 0.00112613, 0.0011982 , 0.00127027, 0.00134234,
       0.00141441, 0.00148649, 0.00155856, 0.00163063, 0.0017027 ,
       0.00177477, 0.00184685, 0.00191892, 0.00199099, 0.00206306,
       0.00213514, 0.00220721, 0.00227928, 0.00235135, 0.00242342,
       0.0024955 , 0.00256757, 0.00263964, 0.00271171, 0.00278378,
       0.00285586, 0.00292793, 0.003     , 0.00307207, 0.00314414,
       0.00321622, 0.00328829, 0.00336036, 0.00343243, 0.0035045 ,
       0.00357658, 0.00364865, 0.00372072, 0.00379279, 0.00386486,
       0.00393694, 0.00400901, 0.00408108, 0.00415315, 0.00422523,
       0.0042973 , 0.00436937, 0.00444144, 0.00451351, 0.00458559]

list_9 = [0.00465766, 0.00472973, 0.0048018 , 0.00487387, 0.00494595,
       0.00501802, 0.00509009, 0.00516216, 0.00523423, 0.00530631,
       0.00537838, 0.00545045, 0.00552252, 0.00559459, 0.00566667,
       0.00573874, 0.00581081, 0.00588288, 0.00595495, 0.00602703,
       0.0060991 , 0.00617117, 0.00624324, 0.00631532, 0.00638739,
       0.00645946, 0.00653153, 0.0066036 , 0.00667568, 0.00674775,
       0.00681982, 0.00689189, 0.00696396, 0.00703604, 0.00710811,
       0.00718018, 0.00725225, 0.00732432, 0.0073964 , 0.00746847,
       0.00754054, 0.00761261, 0.00768468, 0.00775676, 0.00782883,
       0.0079009 , 0.00797297, 0.00804505, 0.00811712, 0.00818919,
       0.00826126, 0.00833333, 0.00840541, 0.00847748, 0.00854955,
       0.00862162, 0.00869369, 0.00876577, 0.00883784, 0.00890991,
       0.00898198, 0.00905405, 0.00912613, 0.0091982 , 0.00927027,
       0.00934234, 0.00941441, 0.00948649, 0.00955856, 0.00963063,
       0.0097027 , 0.00977477, 0.00984685, 0.00991892, 0.00999099,
       0.01063063, 0.01135135, 0.01207207, 0.01279279, 0.01351351,
       0.01423423, 0.01495495, 0.01567568, 0.0163964 , 0.01711712,
       0.01783784, 0.01855856, 0.01927928, 0.02      , 0.02072072,
       0.02144144, 0.02216216, 0.02288288, 0.0236036 , 0.02432432,
       0.02504505, 0.02576577, 0.02648649, 0.02720721, 0.02792793]

list_10 = [0.02864865, 0.02936937, 0.03009009, 0.03081081, 0.03153153,
       0.03225225, 0.03297297, 0.03369369, 0.03441441, 0.03513514,
       0.03585586, 0.03657658, 0.0372973 , 0.03801802, 0.03873874,
       0.03945946, 0.04018018, 0.0409009 , 0.04162162, 0.04234234,
       0.04306306, 0.04378378, 0.0445045 , 0.04522523, 0.04594595,
       0.04666667, 0.04738739, 0.04810811, 0.04882883, 0.04954955,
       0.05027027, 0.05099099, 0.05171171, 0.05243243, 0.05315315,
       0.05387387, 0.05459459, 0.05531532, 0.05603604, 0.05675676,
       0.05747748, 0.0581982 , 0.05891892, 0.05963964, 0.06036036,
       0.06108108, 0.0618018 , 0.06252252, 0.06324324, 0.06396396,
       0.06468468, 0.06540541, 0.06612613, 0.06684685, 0.06756757,
       0.06828829, 0.06900901, 0.06972973, 0.07045045, 0.07117117,
       0.07189189, 0.07261261, 0.07333333, 0.07405405, 0.07477477,
       0.0754955 , 0.07621622, 0.07693694, 0.07765766, 0.07837838,
       0.0790991 , 0.07981982, 0.08054054, 0.08126126, 0.08198198,
       0.0827027 , 0.08342342, 0.08414414, 0.08486486, 0.08558559,
       0.08630631, 0.08702703, 0.08774775, 0.08846847, 0.08918919,
       0.08990991, 0.09063063, 0.09135135, 0.09207207, 0.09279279,
       0.09351351, 0.09423423, 0.09495495, 0.09567568, 0.0963964 ,
       0.09711712, 0.09783784, 0.09855856, 0.09927928, 0.1       ]


list_1_1 = [1.00000000e-09, 1.07207207e-09, 1.14414414e-09, 1.21621622e-09,
       1.28828829e-09, 1.36036036e-09, 1.43243243e-09, 1.50450450e-09,
       1.57657658e-09, 1.64864865e-09, 1.72072072e-09, 1.79279279e-09,
       1.86486486e-09, 1.93693694e-09, 2.00900901e-09, 2.08108108e-09,
       2.15315315e-09, 2.22522523e-09, 2.29729730e-09, 2.36936937e-09,
       2.44144144e-09, 2.51351351e-09, 2.58558559e-09, 2.65765766e-09,
       2.72972973e-09, 2.80180180e-09, 2.87387387e-09, 2.94594595e-09,
       3.01801802e-09, 3.09009009e-09, 3.16216216e-09, 3.23423423e-09,
       3.30630631e-09, 3.37837838e-09, 3.45045045e-09, 3.52252252e-09,
       3.59459459e-09, 3.66666667e-09, 3.73873874e-09, 3.81081081e-09,
       3.88288288e-09, 3.95495495e-09, 4.02702703e-09, 4.09909910e-09,
       4.17117117e-09, 4.24324324e-09, 4.31531532e-09, 4.38738739e-09]
list_1_2 = [4.45945946e-09, 4.53153153e-09, 4.60360360e-09, 4.67567568e-09,
       4.74774775e-09, 4.81981982e-09, 4.89189189e-09, 4.96396396e-09,
       5.03603604e-09, 5.10810811e-09, 5.18018018e-09, 5.25225225e-09,
       5.32432432e-09, 5.39639640e-09, 5.46846847e-09, 5.54054054e-09,
       5.61261261e-09, 5.68468468e-09, 5.75675676e-09, 5.82882883e-09,
       5.90090090e-09, 5.97297297e-09, 6.04504505e-09, 6.11711712e-09,
       6.18918919e-09, 6.26126126e-09, 6.33333333e-09, 6.40540541e-09,
       6.47747748e-09, 6.54954955e-09, 6.62162162e-09, 6.69369369e-09,
       6.76576577e-09, 6.83783784e-09, 6.90990991e-09, 6.98198198e-09,
       7.05405405e-09, 7.12612613e-09, 7.19819820e-09, 7.27027027e-09,
       7.34234234e-09, 7.41441441e-09, 7.48648649e-09, 7.55855856e-09,
       7.63063063e-09, 7.70270270e-09, 7.77477477e-09, 7.84684685e-09,
       7.91891892e-09, 7.99099099e-09, 8.06306306e-09, 8.13513514e-09]

list_2_1 = [8.20720721e-09, 8.27927928e-09, 8.35135135e-09, 8.42342342e-09,
       8.49549550e-09, 8.56756757e-09, 8.63963964e-09, 8.71171171e-09,
       8.78378378e-09, 8.85585586e-09, 8.92792793e-09, 9.00000000e-09,
       9.07207207e-09, 9.14414414e-09, 9.21621622e-09, 9.28828829e-09,
       9.36036036e-09, 9.43243243e-09, 9.50450450e-09, 9.57657658e-09,
       9.64864865e-09, 9.72072072e-09, 9.79279279e-09, 9.86486486e-09,
       9.93693694e-09, 1.00900901e-08, 1.08108108e-08, 1.15315315e-08,
       1.22522523e-08, 1.29729730e-08, 1.36936937e-08, 1.44144144e-08,
       1.51351351e-08, 1.58558559e-08, 1.65765766e-08, 1.72972973e-08,
       1.80180180e-08, 1.87387387e-08, 1.94594595e-08, 2.01801802e-08,
       2.09009009e-08, 2.16216216e-08, 2.23423423e-08, 2.30630631e-08,
       2.37837838e-08, 2.45045045e-08, 2.52252252e-08, 2.59459459e-08]
list_2_2 = [2.66666667e-08, 2.73873874e-08, 2.81081081e-08, 2.88288288e-08,
       2.95495495e-08, 3.02702703e-08, 3.09909910e-08, 3.17117117e-08,
       3.24324324e-08, 3.31531532e-08, 3.38738739e-08, 3.45945946e-08,
       3.53153153e-08, 3.60360360e-08, 3.67567568e-08, 3.74774775e-08,
       3.81981982e-08, 3.89189189e-08, 3.96396396e-08, 4.03603604e-08,
       4.10810811e-08, 4.18018018e-08, 4.25225225e-08, 4.32432432e-08,
       4.39639640e-08, 4.46846847e-08, 4.54054054e-08, 4.61261261e-08,
       4.68468468e-08, 4.75675676e-08, 4.82882883e-08, 4.90090090e-08,
       4.97297297e-08, 5.04504505e-08, 5.11711712e-08, 5.18918919e-08,
       5.26126126e-08, 5.33333333e-08, 5.40540541e-08, 5.47747748e-08,
       5.54954955e-08, 5.62162162e-08, 5.69369369e-08, 5.76576577e-08,
       5.83783784e-08, 5.90990991e-08, 5.98198198e-08, 6.05405405e-08,
       6.12612613e-08, 6.19819820e-08, 6.27027027e-08, 6.34234234e-08]

list_3_1 = [6.41441441e-08, 6.48648649e-08, 6.55855856e-08, 6.63063063e-08,
       6.70270270e-08, 6.77477477e-08, 6.84684685e-08, 6.91891892e-08,
       6.99099099e-08, 7.06306306e-08, 7.13513514e-08, 7.20720721e-08,
       7.27927928e-08, 7.35135135e-08, 7.42342342e-08, 7.49549550e-08,
       7.56756757e-08, 7.63963964e-08, 7.71171171e-08, 7.78378378e-08,
       7.85585586e-08, 7.92792793e-08, 8.00000000e-08, 8.07207207e-08,
       8.14414414e-08, 8.21621622e-08, 8.28828829e-08, 8.36036036e-08,
       8.43243243e-08, 8.50450450e-08, 8.57657658e-08, 8.64864865e-08,
       8.72072072e-08, 8.79279279e-08, 8.86486486e-08, 8.93693694e-08,
       9.00900901e-08, 9.08108108e-08, 9.15315315e-08, 9.22522523e-08,
       9.29729730e-08, 9.36936937e-08, 9.44144144e-08, 9.51351351e-08,
       9.58558559e-08, 9.65765766e-08, 9.72972973e-08, 9.80180180e-08]
list_3_2 = [9.87387387e-08, 9.94594595e-08, 1.01801802e-07, 1.09009009e-07,
       1.16216216e-07, 1.23423423e-07, 1.30630631e-07, 1.37837838e-07,
       1.45045045e-07, 1.52252252e-07, 1.59459459e-07, 1.66666667e-07,
       1.73873874e-07, 1.81081081e-07, 1.88288288e-07, 1.95495495e-07,
       2.02702703e-07, 2.09909910e-07, 2.17117117e-07, 2.24324324e-07,
       2.31531532e-07, 2.38738739e-07, 2.45945946e-07, 2.53153153e-07,
       2.60360360e-07, 2.67567568e-07, 2.74774775e-07, 2.81981982e-07,
       2.89189189e-07, 2.96396396e-07, 3.03603604e-07, 3.10810811e-07,
       3.18018018e-07, 3.25225225e-07, 3.32432432e-07, 3.39639640e-07,
       3.46846847e-07, 3.54054054e-07, 3.61261261e-07, 3.68468468e-07,
       3.75675676e-07, 3.82882883e-07, 3.90090090e-07, 3.97297297e-07,
       4.04504505e-07, 4.11711712e-07, 4.18918919e-07, 4.26126126e-07,
       4.33333333e-07, 4.40540541e-07, 4.47747748e-07, 4.54954955e-07]

list_4_1 = [4.62162162e-07, 4.69369369e-07, 4.76576577e-07, 4.83783784e-07,
       4.90990991e-07, 4.98198198e-07, 5.05405405e-07, 5.12612613e-07,
       5.19819820e-07, 5.27027027e-07, 5.34234234e-07, 5.41441441e-07,
       5.48648649e-07, 5.55855856e-07, 5.63063063e-07, 5.70270270e-07,
       5.77477477e-07, 5.84684685e-07, 5.91891892e-07, 5.99099099e-07,
       6.06306306e-07, 6.13513514e-07, 6.20720721e-07, 6.27927928e-07,
       6.35135135e-07, 6.42342342e-07, 6.49549550e-07, 6.56756757e-07,
       6.63963964e-07, 6.71171171e-07, 6.78378378e-07, 6.85585586e-07,
       6.92792793e-07, 7.00000000e-07, 7.07207207e-07, 7.14414414e-07,
       7.21621622e-07, 7.28828829e-07, 7.36036036e-07, 7.43243243e-07,
       7.50450450e-07, 7.57657658e-07, 7.64864865e-07, 7.72072072e-07,
       7.79279279e-07, 7.86486486e-07, 7.93693694e-07, 8.00900901e-07]
            
list_4_2 = [8.08108108e-07, 8.15315315e-07, 8.22522523e-07, 8.29729730e-07,
       8.36936937e-07, 8.44144144e-07, 8.51351351e-07, 8.58558559e-07,
       8.65765766e-07, 8.72972973e-07, 8.80180180e-07, 8.87387387e-07,
       8.94594595e-07, 9.01801802e-07, 9.09009009e-07, 9.16216216e-07,
       9.23423423e-07, 9.30630631e-07, 9.37837838e-07, 9.45045045e-07,
       9.52252252e-07, 9.59459459e-07, 9.66666667e-07, 9.73873874e-07,
       9.81081081e-07, 9.88288288e-07, 9.95495495e-07, 1.02702703e-06,
       1.09909910e-06, 1.17117117e-06, 1.24324324e-06, 1.31531532e-06,
       1.38738739e-06, 1.45945946e-06, 1.53153153e-06, 1.60360360e-06,
       1.67567568e-06, 1.74774775e-06, 1.81981982e-06, 1.89189189e-06,
       1.96396396e-06, 2.03603604e-06, 2.10810811e-06, 2.18018018e-06,
       2.25225225e-06, 2.32432432e-06, 2.39639640e-06, 2.46846847e-06,
       2.54054054e-06, 2.61261261e-06, 2.68468468e-06, 2.75675676e-06]

list_5_1 = [2.82882883e-06, 2.90090090e-06, 2.97297297e-06, 3.04504505e-06,
       3.11711712e-06, 3.18918919e-06, 3.26126126e-06, 3.33333333e-06,
       3.40540541e-06, 3.47747748e-06, 3.54954955e-06, 3.62162162e-06,
       3.69369369e-06, 3.76576577e-06, 3.83783784e-06, 3.90990991e-06,
       3.98198198e-06, 4.05405405e-06, 4.12612613e-06, 4.19819820e-06,
       4.27027027e-06, 4.34234234e-06, 4.41441441e-06, 4.48648649e-06,
       4.55855856e-06, 4.63063063e-06, 4.70270270e-06, 4.77477477e-06,
       4.84684685e-06, 4.91891892e-06, 4.99099099e-06, 5.06306306e-06,
       5.13513514e-06, 5.20720721e-06, 5.27927928e-06, 5.35135135e-06,
       5.42342342e-06, 5.49549550e-06, 5.56756757e-06, 5.63963964e-06,
       5.71171171e-06, 5.78378378e-06, 5.85585586e-06, 5.92792793e-06,
       6.00000000e-06, 6.07207207e-06, 6.14414414e-06, 6.21621622e-06]
list_5_2 = [6.28828829e-06, 6.36036036e-06, 6.43243243e-06, 6.50450450e-06,
       6.57657658e-06, 6.64864865e-06, 6.72072072e-06, 6.79279279e-06,
       6.86486486e-06, 6.93693694e-06, 7.00900901e-06, 7.08108108e-06,
       7.15315315e-06, 7.22522523e-06, 7.29729730e-06, 7.36936937e-06,
       7.44144144e-06, 7.51351351e-06, 7.58558559e-06, 7.65765766e-06,
       7.72972973e-06, 7.80180180e-06, 7.87387387e-06, 7.94594595e-06,
       8.01801802e-06, 8.09009009e-06, 8.16216216e-06, 8.23423423e-06,
       8.30630631e-06, 8.37837838e-06, 8.45045045e-06, 8.52252252e-06,
       8.59459459e-06, 8.66666667e-06, 8.73873874e-06, 8.81081081e-06,
       8.88288288e-06, 8.95495495e-06, 9.02702703e-06, 9.09909910e-06,
       9.17117117e-06, 9.24324324e-06, 9.31531532e-06, 9.38738739e-06,
       9.45945946e-06, 9.53153153e-06, 9.60360360e-06, 9.67567568e-06,
       9.74774775e-06, 9.81981982e-06, 9.89189189e-06, 9.96396396e-06]

list_6_1 = [1.03603604e-05, 1.10810811e-05, 1.18018018e-05, 1.25225225e-05,
       1.32432432e-05, 1.39639640e-05, 1.46846847e-05, 1.54054054e-05,
       1.61261261e-05, 1.68468468e-05, 1.75675676e-05, 1.82882883e-05,
       1.90090090e-05, 1.97297297e-05, 2.04504505e-05, 2.11711712e-05,
       2.18918919e-05, 2.26126126e-05, 2.33333333e-05, 2.40540541e-05,
       2.47747748e-05, 2.54954955e-05, 2.62162162e-05, 2.69369369e-05,
       2.76576577e-05, 2.83783784e-05, 2.90990991e-05, 2.98198198e-05,
       3.05405405e-05, 3.12612613e-05, 3.19819820e-05, 3.27027027e-05,
       3.34234234e-05, 3.41441441e-05, 3.48648649e-05, 3.55855856e-05,
       3.63063063e-05, 3.70270270e-05, 3.77477477e-05, 3.84684685e-05,
       3.91891892e-05, 3.99099099e-05, 4.06306306e-05, 4.13513514e-05,
       4.20720721e-05, 4.27927928e-05, 4.35135135e-05, 4.42342342e-05]
list_6_2 = [4.49549550e-05, 4.56756757e-05, 4.63963964e-05, 4.71171171e-05,
       4.78378378e-05, 4.85585586e-05, 4.92792793e-05, 5.00000000e-05,
       5.07207207e-05, 5.14414414e-05, 5.21621622e-05, 5.28828829e-05,
       5.36036036e-05, 5.43243243e-05, 5.50450450e-05, 5.57657658e-05,
       5.64864865e-05, 5.72072072e-05, 5.79279279e-05, 5.86486486e-05,
       5.93693694e-05, 6.00900901e-05, 6.08108108e-05, 6.15315315e-05,
       6.22522523e-05, 6.29729730e-05, 6.36936937e-05, 6.44144144e-05,
       6.51351351e-05, 6.58558559e-05, 6.65765766e-05, 6.72972973e-05,
       6.80180180e-05, 6.87387387e-05, 6.94594595e-05, 7.01801802e-05,
       7.09009009e-05, 7.16216216e-05, 7.23423423e-05, 7.30630631e-05,
       7.37837838e-05, 7.45045045e-05, 7.52252252e-05, 7.59459459e-05,
       7.66666667e-05, 7.73873874e-05, 7.81081081e-05, 7.88288288e-05,
       7.95495495e-05, 8.02702703e-05, 8.09909910e-05, 8.17117117e-05]

list_7_1 = [8.24324324e-05, 8.31531532e-05, 8.38738739e-05, 8.45945946e-05,
       8.53153153e-05, 8.60360360e-05, 8.67567568e-05, 8.74774775e-05,
       8.81981982e-05, 8.89189189e-05, 8.96396396e-05, 9.03603604e-05,
       9.10810811e-05, 9.18018018e-05, 9.25225225e-05, 9.32432432e-05,
       9.39639640e-05, 9.46846847e-05, 9.54054054e-05, 9.61261261e-05,
       9.68468468e-05, 9.75675676e-05, 9.82882883e-05, 9.90090090e-05,
       9.97297297e-05, 1.04504505e-04, 1.11711712e-04, 1.18918919e-04,
       1.26126126e-04, 1.33333333e-04, 1.40540541e-04, 1.47747748e-04,
       1.54954955e-04, 1.62162162e-04, 1.69369369e-04, 1.76576577e-04,
       1.83783784e-04, 1.90990991e-04, 1.98198198e-04, 2.05405405e-04,
       2.12612613e-04, 2.19819820e-04, 2.27027027e-04, 2.34234234e-04,
       2.41441441e-04, 2.48648649e-04, 2.55855856e-04, 2.63063063e-04,]
list_7_2 = [2.70270270e-04, 2.77477477e-04, 2.84684685e-04, 2.91891892e-04,
       2.99099099e-04, 3.06306306e-04, 3.13513514e-04, 3.20720721e-04,
       3.27927928e-04, 3.35135135e-04, 3.42342342e-04, 3.49549550e-04,
       3.56756757e-04, 3.63963964e-04, 3.71171171e-04, 3.78378378e-04,
       3.85585586e-04, 3.92792793e-04, 4.00000000e-04, 4.07207207e-04,
       4.14414414e-04, 4.21621622e-04, 4.28828829e-04, 4.36036036e-04,
       4.43243243e-04, 4.50450450e-04, 4.57657658e-04, 4.64864865e-04,
       4.72072072e-04, 4.79279279e-04, 4.86486486e-04, 4.93693694e-04,
       5.00900901e-04, 5.08108108e-04, 5.15315315e-04, 5.22522523e-04,
       5.29729730e-04, 5.36936937e-04, 5.44144144e-04, 5.51351351e-04,
       5.58558559e-04, 5.65765766e-04, 5.72972973e-04, 5.80180180e-04,
       5.87387387e-04, 5.94594595e-04, 6.01801802e-04, 6.09009009e-04,
       6.16216216e-04, 6.23423423e-04, 6.30630631e-04, 6.37837838e-04]

list_8_1 = [0.00064505, 0.00065225, 0.00065946, 0.00066667, 0.00067387,
       0.00068108, 0.00068829, 0.0006955 , 0.0007027 , 0.00070991,
       0.00071712, 0.00072432, 0.00073153, 0.00073874, 0.00074595,
       0.00075315, 0.00076036, 0.00076757, 0.00077477, 0.00078198,
       0.00078919, 0.0007964 , 0.0008036 , 0.00081081, 0.00081802,
       0.00082523, 0.00083243, 0.00083964, 0.00084685, 0.00085405,
       0.00086126, 0.00086847, 0.00087568, 0.00088288, 0.00089009,
       0.0008973 , 0.0009045 , 0.00091171, 0.00091892, 0.00092613,
       0.00093333, 0.00094054, 0.00094775, 0.00095495, 0.00096216,
       0.00096937, 0.00097658, 0.00098378, 0.00099099, 0.0009982]
list_8_2 = [0.00105405, 0.00112613, 0.0011982 , 0.00127027, 0.00134234,
       0.00141441, 0.00148649, 0.00155856, 0.00163063, 0.0017027 ,
       0.00177477, 0.00184685, 0.00191892, 0.00199099, 0.00206306,
       0.00213514, 0.00220721, 0.00227928, 0.00235135, 0.00242342,
       0.0024955 , 0.00256757, 0.00263964, 0.00271171, 0.00278378,
       0.00285586, 0.00292793, 0.003     , 0.00307207, 0.00314414,
       0.00321622, 0.00328829, 0.00336036, 0.00343243, 0.0035045 ,
       0.00357658, 0.00364865, 0.00372072, 0.00379279, 0.00386486,
       0.00393694, 0.00400901, 0.00408108, 0.00415315, 0.00422523,
       0.0042973 , 0.00436937, 0.00444144, 0.00451351, 0.00458559]

list_9_1 = [0.00465766, 0.00472973, 0.0048018 , 0.00487387, 0.00494595,
       0.00501802, 0.00509009, 0.00516216, 0.00523423, 0.00530631,
       0.00537838, 0.00545045, 0.00552252, 0.00559459, 0.00566667,
       0.00573874, 0.00581081, 0.00588288, 0.00595495, 0.00602703,
       0.0060991 , 0.00617117, 0.00624324, 0.00631532, 0.00638739,
       0.00645946, 0.00653153, 0.0066036 , 0.00667568, 0.00674775,
       0.00681982, 0.00689189, 0.00696396, 0.00703604, 0.00710811,
       0.00718018, 0.00725225, 0.00732432, 0.0073964 , 0.00746847,
       0.00754054, 0.00761261, 0.00768468, 0.00775676, 0.00782883,
       0.0079009 , 0.00797297, 0.00804505, 0.00811712, 0.00818919]
list_9_2 = [0.00826126, 0.00833333, 0.00840541, 0.00847748, 0.00854955,
       0.00862162, 0.00869369, 0.00876577, 0.00883784, 0.00890991,
       0.00898198, 0.00905405, 0.00912613, 0.0091982 , 0.00927027,
       0.00934234, 0.00941441, 0.00948649, 0.00955856, 0.00963063,
       0.0097027 , 0.00977477, 0.00984685, 0.00991892, 0.00999099,
       0.01063063, 0.01135135, 0.01207207, 0.01279279, 0.01351351,
       0.01423423, 0.01495495, 0.01567568, 0.0163964 , 0.01711712,
       0.01783784, 0.01855856, 0.01927928, 0.02      , 0.02072072,
       0.02144144, 0.02216216, 0.02288288, 0.0236036 , 0.02432432,
       0.02504505, 0.02576577, 0.02648649, 0.02720721, 0.02792793]

list_10_1 = [0.02864865, 0.02936937, 0.03009009, 0.03081081, 0.03153153,
       0.03225225, 0.03297297, 0.03369369, 0.03441441, 0.03513514,
       0.03585586, 0.03657658, 0.0372973 , 0.03801802, 0.03873874,
       0.03945946, 0.04018018, 0.0409009 , 0.04162162, 0.04234234,
       0.04306306, 0.04378378, 0.0445045 , 0.04522523, 0.04594595,
       0.04666667, 0.04738739, 0.04810811, 0.04882883, 0.04954955,
       0.05027027, 0.05099099, 0.05171171, 0.05243243, 0.05315315,
       0.05387387, 0.05459459, 0.05531532, 0.05603604, 0.05675676,
       0.05747748, 0.0581982 , 0.05891892, 0.05963964, 0.06036036,
       0.06108108, 0.0618018 , 0.06252252, 0.06324324, 0.06396396]
           
      
list_10_2 = [0.06468468, 0.06540541, 0.06612613, 0.06684685, 0.06756757,
       0.06828829, 0.06900901, 0.06972973, 0.07045045, 0.07117117,
       0.07189189, 0.07261261, 0.07333333, 0.07405405, 0.07477477,
       0.0754955 , 0.07621622, 0.07693694, 0.07765766, 0.07837838,
       0.0790991 , 0.07981982, 0.08054054, 0.08126126, 0.08198198,
       0.0827027 , 0.08342342, 0.08414414, 0.08486486, 0.08558559,
       0.08630631, 0.08702703, 0.08774775, 0.08846847, 0.08918919,
       0.08990991, 0.09063063, 0.09135135, 0.09207207, 0.09279279,
       0.09351351, 0.09423423, 0.09495495, 0.09567568, 0.0963964 ,
       0.09711712, 0.09783784, 0.09855856, 0.09927928, 0.1       ]