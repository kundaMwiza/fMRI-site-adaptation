function target_6_TPE_5_1000_1000
clear;
% input_file = '/shared/tale2/Shared/data/abide/functionals/cpac/filt_noglobal/cc200_TPE.mat';
input_file = '/media/shuo/MyDrive/data/brain/ABIDE_pcp/cpac/filt_noglobal/cc200_TPE.mat';
load(input_file, 'connectivity')
load('/media/shuo/MyDrive/data/brain/ABIDE_pcp/cpac/filt_noglobal/site_label.mat', 'site_label')
target = 6;
alpha = 1;
beta = 1;
dim = 5;
n_sites = length(unique(site_label));
S = cell(1, n_sites - 1);
iter_s = 1;
for i=0:n_sites-1
    if i == target
        T = connectivity(site_label==i, :).';
    else
        S{1, iter_s} = connectivity(site_label==i, :).';
        iter_s = iter_s + 1;
    end
end
[Z,Ez,Ew,W,Wi] = maLRR(T, S, dim, 2, alpha, beta);
fname = '/media/shuo/MyDrive/data/brain/ABIDE_pcp/cpac/filt_noglobal/target_6_TPE_5_1_1_malrr.mat';
save(fname, 'Z','Ez','Ew','W','Wi')
