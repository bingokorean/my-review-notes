
filename = 'ex3data1.mat';


data = load(filename,'-mat');
f = fieldnames(data);
for k=1; size(f,1)
    xlswrite('1.xlsx', data.(f{k}), f{k});
end