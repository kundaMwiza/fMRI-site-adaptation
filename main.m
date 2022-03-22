clear;

kind = "TPE";
input_file = "/media/shuo/MyDrive/data/brain/ABIDE_pcp/cpac/filt_noglobal/cc200_" + kind + ".mat";
load(input_file)
load("/media/shuo/MyDrive/data/brain/ABIDE_pcp/cpac/filt_noglobal/site_label.mat")

% [X] = pca(connectivity, 'NumComponents',1000);

alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
betas = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
dims = [3, 5, 7, 9, 11, 13, 15];

% target = 0;
n_sites = length(unique(site_label));
S = cell(1, n_sites - 1);
iter_s = 1;

for target=0:n_sites-1
    for i=0:n_sites-1   
        if i == target
            T = connectivity(site_label==i, :).';
        else
            S{1, iter_s} = connectivity(site_label==i, :).';
            iter_s = iter_s + 1;
        end
    end
    for iter_alpha=1:length(alphas)
        alpha = alphas(iter_alpha);
        for iter_beta=1:length(betas)
            beta = betas(iter_beta);
            for iter_dim=1:length(dims)
                dim = dims(iter_dim);
                [Z,Ez,Ew,W,Wi] = maLRR(T, S, dim, 50, alpha, beta);
                fname = "/media/shuo/MyDrive/data/brain/ABIDE_pcp/cpac/filt_noglobal/" + "target_" + target + "_" + kind + "_" + alpha + "_" + beta + "_malrr.mat";
                save(fname, Z)
            end
        end        
    end
end
