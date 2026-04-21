#!/usr/bin/env python
import os,sys

print('Start\n')
########   customization  area #########
queue = 'longlunch' # give bsub queue -- 8nm (8 minutes), 1nh (1 hour), 8nh, 1nd (1day), 2nd, 1nw (1 week), 2nw
WorkDir = '/afs/cern.ch/user/e/eallergu/workdir/GNN/tmp/EdgeClassifier/Training'

ConfigPath = '/eos/user/e/eallergu/GNN/L1Nano_GenParticlePropagator/CMSSW_15_1_0_pre4/src/emeleTrigger/configs/'
ConfigFile = 'training_edge_classification.yml'

########   customization end   #########
path = os.getcwd()
print('Do not worry about folder creation:\n')
os.system("rm -rf %s" %(WorkDir))
os.system("mkdir %s" %(WorkDir))
os.system("mkdir %s/exec" %(WorkDir))
os.system("mkdir %s/batchlogs" %(WorkDir))

config_name = ConfigFile.split('.')[0]

##### creating job #####
with open("%s/exec/job_%s.sh" %(WorkDir, config_name), 'w') as fout:
    fout.write("#!/bin/sh\n")
    fout.write("echo\n")
    fout.write("echo\n")
    fout.write("echo 'START---------------'\n")
    fout.write("echo 'Work Directory ' ${PWD}\n")
    fout.write("cd "+str(path)+"\n")
    fout.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh\n")
    fout.write("python TrainEdgeClassificationFromGraph.py --config %s --do_train\n" %(ConfigPath+ConfigFile))  
    fout.write("echo 'STOP---------------'\n")
    fout.write("echo\n")
    fout.write("echo\n")
os.system("chmod 755 %s/exec/job_%s.sh" %(WorkDir, config_name))

###### create submit.sub file ####
with open('%s/submit.sub' %(WorkDir), 'w') as fout:
    fout.write("executable              = $(filename)\n")
    fout.write("arguments               = $(ClusterId)$(ProcId)\n")
    fout.write("output                  = %s/batchlogs/$(ClusterId).$(ProcId).out\n" %(WorkDir))
    fout.write("error                   = %s/batchlogs/$(ClusterId).$(ProcId).err\n"    %(WorkDir))
    fout.write("log                     = %s/batchlogs/$(ClusterId).log\n"             %(WorkDir))
    fout.write("request_gpus            = 1\n")
    fout.write('+JobFlavour = "%s"\n' %(queue))
    fout.write("\n")
    fout.write("queue filename matching (%s/exec/job_*sh)\n" %(WorkDir))

###### sends bjobs ######
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
