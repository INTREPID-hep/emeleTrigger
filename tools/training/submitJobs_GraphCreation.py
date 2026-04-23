#!/usr/bin/env python
import os,sys
import glob
import time
import json

print('Start\n')
########   Customization  area   #########
queue      = 1                                                                               # Number of instances of the job to be sent (default = 1). (¿Se está usando?)
JobFlavour = 'longlunch'                                                                     # e.g. expresso, microcentury, longlunch, workday, tomorrow -> up to 48 hours  (¿Se puede usar o es solo CERN?)

NJobs = 8                                                                                    # Number of jobs to be sent (maximum = 8).

WorkDir = '/lhome/ext/uovi156/uovi1564/workdir/S5'                                           # Work directory (path) where the .log, .out and .err files will be stored.

ConfigPath = '/lhome/ext/uovi156/uovi1564/emeleTrigger/configs/'                             # Path to the configuration file.
ConfigFile = 'config_PelayoFiles.yml'                                                        # Configuration file name.

MaxEvents = 150000                                                                           # Maximum number of events to be processed.
MaxFiles  = None                                                                             # Maximum number of files to be processed (None for all files).

InputPath  = '/lustre/ific.uv.es/ml/uovi156/data/prod/S5/'                                   # Path to the input files.
InputFiles = 'omtf_nano_*.root'                                                              # Input ROOT files name (or name structure).
TreeName   = 'Events'                                                                        # Name of the tree

OutputPath   = '/lustre/ific.uv.es/ml/uovi156/data/graphs_dataset/l1nano_graphs_prod/S5/'    # Output path to store the result (.pt) files.
OutputPrefix = 'l1nano_dataset_'                                                             # Prefix for the output files (1 file per job).
########   Customization end   #########


########   Output, work and input directories   #########
path = os.getcwd()
print('Do not worry about folder creation.\n')


### Output directory
if not os.path.exists(OutputPath):
    print("Output Directory {} does not exist.".format(OutputPath))
    os.system("mkdir {}".format(OutputPath))
else :
    print("Warning: Output Directory already exists. It will be overwritten.\n")
    print("Output Directory: {}".format(OutputPath))


### Work directory
print('The Work Directory will be overwritten.\n')
os.system("rm -rf %s" %(WorkDir))
os.system("mkdir %s" %(WorkDir))
os.system("mkdir %s/exec" %(WorkDir))
os.system("mkdir %s/batchlogs" %(WorkDir))


### Input files
input_name  = os.path.join(InputPath, InputFiles)
input_files = glob.glob(input_name)
n_infiles   = len(input_files)
print("Input files: {} (Files: {})".format(n_infiles, input_name))
if MaxFiles == None:
    print("Maximum number of files to be processed: {}".format(n_infiles))
    infiles_batch = n_infiles // NJobs
    infiles_extra = n_infiles % NJobs              # Some batches will have an extra file (rounded)
else:
    print("Maximum number of files to be processed: {}".format(MaxFiles))
    if n_infiles > MaxFiles:
        infiles_batch = MaxFiles // NJobs
        infiles_extra = MaxFiles % NJobs           # Some batches will have an extra file (rounded)
    else:
        infiles_batch = n_infiles // NJobs
        infiles_extra = n_infiles % NJobs          # Some batches will have an extra file (rounded)
print("Running {} jobs with {} input files each (+-1 file).".format(NJobs, infiles_batch))
print("{} jobs will process {} files.".format(infiles_extra, infiles_batch+1))

batches      = {}
initial_size = 0

for job_idx in range(NJobs):
    size       = infiles_batch + (1 if job_idx < infiles_extra else 0)
    batch_name = f"{job_idx+1}"
    batches[batch_name] = json.dumps(input_files[initial_size:initial_size + size])
    initial_size += size
#print("Batches: {}".format(batches))


########   Creating job   #########
config_name = ConfigFile.split('.')[0]
for key, batch in batches.items():
    output_file = os.path.join(OutputPath, OutputPrefix + key + '.pt')
    with open("%s/exec/job_%s_%s.sh" %(WorkDir, key, config_name), 'w') as fout:
        fout.write("#!/bin/sh\n")
        fout.write("echo\n")
        fout.write("echo\n")
        fout.write("echo 'START---------------'\n")
        fout.write("start=$(date +%s)\n")
        fout.write("echo 'Work Directory ' ${PWD}\n")
        fout.write("cd "+str(path)+"\n")
        fout.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh\n")
        fout.write("python InputDataset.py --config '%s' --root_dir '%s' --tree_name '%s' --max_files %d --max_events %d --save_path '%s' --do_batches --debug\n" %(ConfigPath+ConfigFile, batch, TreeName, n_infiles, MaxEvents, output_file))
        #fout.write("./run_l1nano_save_dataset.sh '%s' %d %d '%s' --config '%s'\n" %(input_name, MaxEvents, MaxFiles, OutputPath+OutputPrefix+'.pt', ConfigPath+ConfigFile))
        fout.write("end=$(date +%s)\n")
        fout.write('echo "Total time: $((end - start)) seconds"\n')
        fout.write("echo 'STOP---------------'\n")
        fout.write("echo\n")
        fout.write("echo\n")
    os.system("chmod 755 %s/exec/job_%s_%s.sh" %(WorkDir, key, config_name))


########   Creating submit.sub file   #########
with open('%s/submit.sub' %(WorkDir), 'w') as fout:
    fout.write("executable              = $(filename)\n")
    fout.write("arguments               = $(Cluster)$(Process)\n")
    fout.write("output                  = %s/batchlogs/$(Cluster).$(Process).out\n" %(WorkDir))
    fout.write("error                   = %s/batchlogs/$(Cluster).$(Process).err\n" %(WorkDir))
    fout.write("log                     = %s/batchlogs/$(Cluster).log\n"            %(WorkDir))
    #fout.write("request_gpus            = 1\n")
    fout.write("\n")
    fout.write("queue filename matching (%s/exec/job_*sh)\n" %(WorkDir))


########   Send bjobs   #########
print()
print("### To submit all jobs do: ")
print("....................................................................")
print("cd %s" %(WorkDir))
print("cat submit.sub")
print("condor_submit submit.sub")
print("cd -")

print()
print("### Check your jobs:")
print("condor_q")
print()
print("....................................................................")
print('End')
print()
