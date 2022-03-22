# kinds = ["TPE", "tangent"]
kinds = ["TPE"]
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
betas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
dims = [5, 10, 20, 50]


def gen_job_srp(target, kind, dim, alpha, beta):
    srp_fname = "scripts/target_%s_%s_%s_%s_%s.m" % (target, kind, dim, alpha, beta)
    srp = open(srp_fname, "w")
    srp.write("clear;\n")
    srp.write("input_file = '/shared/tale2/Shared/data/abide/functionals/cpac/"
              "filt_noglobal/cc200_%s.mat';\n" % kind)
    srp.write("load(input_file)\n")
    srp.write("load('/shared/tale2/Shared/data/abide/functionals/cpac/filt_noglobal/site_label.mat')\n")
    srp.write("target = %s;\n" % target)
    srp.write("alpha = %s;\n" % alpha)
    srp.write("beta = %s;\n" % beta)
    srp.write("dim = %s;\n" % dim)
    srp.write("n_sites = length(unique(site_label));\n")
    srp.write("S = cell(1, n_sites - 1);\n")
    srp.write("iter_s = 1;\n")
    srp.write("for i=0:n_sites-1\n")
    srp.write("    if i == target\n")
    srp.write("        T = connectivity(site_label==i, :).';\n")
    srp.write("    else\n")
    srp.write("        S{1, iter_s} = connectivity(site_label==i, :).';\n")
    srp.write("        iter_s = iter_s + 1;\n")
    srp.write("    end\n")
    srp.write("end\n")
    srp.write("[Z,Ez,Ew,W,Wi] = maLRR(T, S, dim, 50, alpha, beta);\n")
    srp.write("fname = '/shared/tale2/Shared/data/abide/functionals/cpac/filt_noglobal/"
              "target_%s_%s_%s_%s_%s_malrr.mat;\n" % (target, kind, dim, alpha, beta))
    srp.write("save(fname, 'Z','z','Ew','W','Wi')\n")
    srp.close()

    job_fname = "scripts/target_%s_%s_%s_%s_%s.sge" % (target, kind, dim, alpha, beta)
    job_file = open(job_fname, "w")
    job_file.write("#!/bin/bash\n")
    job_file.write("# $ -P tale\n")
    job_file.write("# $ -q tale.q\n")
    job_file.write("#$ -l rmem=16G\n")
    job_file.write("module load apps/matlab/2021b/binary\n")
    job_file.write("\n")
    job_file.write("cd /shared/tale2/Shared/szhou/code/multi-site-fmri/")
    job_file.write("\n")
    job_file.write("matlab -nodesktop -nosplash -r %s\n" % srp_fname)
    job_file.close()

    return job_fname


qsub_fname = "scripts/qsub.sh"
qsub_file = open(qsub_fname, "w")
for target in range(20):
    for kind in kinds:
        for dim in dims:
            job_fname = gen_job_srp(target, kind, dim, 1, 1)
            qsub_file.write("qsub %s\n" % job_fname)
        for alpha in alphas:
            job_fname = gen_job_srp(target, kind, 100, alpha, 1)
            qsub_file.write("qsub %s\n" % job_fname)
        for beta in betas:
            job_fname = gen_job_srp(target, kind, 100, 1, beta)
            qsub_file.write("qsub %s\n" % job_fname)

qsub_file.close()
