import splitfolders

input_folder = 'D:\Sign-Language-Recognition--For-Marathi-Language-main\skinmask_marathi_dataset'

splitfolders.ratio(input_folder, output="D:/New",
                   seed=42, ratio=(.8,.0,.2),
                   group_prefix=None)