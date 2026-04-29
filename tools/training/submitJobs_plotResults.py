#!/usr/bin/env python
import os,sys

print('Start\n')
########   customization  area #########
queue      = 1                                                                               # Number of instances of the job to be sent (default = 1). (¿Se está usando?)
JobFlavour = 'longlunch'                                                                     # e.g. expresso, microcentury, longlunch, workday, tomorrow -> up to 48 hours  (¿Se puede usar o es solo CERN?)

WorkDir = '/lhome/ext/uovi156/uovi1564/workdir/Plotter/SAGE'                                 # Work directory (path) where the .log, .out and .err files will be stored.

ConfigPath = '/lhome/ext/uovi156/uovi1564/emeleTrigger/configs/'                             # Path to the configuration file.
ConfigFile = 'training_edge_classification_quick.yml'

JustPrint = False
########   customization end   #########


########   Work directory   #########
path = os.getcwd()
print('Do not worry about folder creation:\n')


### Work directory
print('The Work Directory will be overwritten.\n')
os.system("rm -rf %s" %(WorkDir))
os.system("mkdir %s" %(WorkDir))
os.system("mkdir %s/exec" %(WorkDir))
os.system("mkdir %s/batchlogs" %(WorkDir))


########   Just print   #########
if JustPrint:
    print("##########################")
    print("source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh\n")
    print("python TrainEdgeClassificationFromGraph.py --config %s --do_validation --do_test\n" %(ConfigPath+ConfigFile))
    print("##########################")
    sys.exit()

########   Creating job   #########
config_name = ConfigFile.split('.')[0]
with open("%s/exec/job_%s.sh" %(WorkDir, config_name), 'w') as fout:
    fout.write("#!/bin/sh\n")
    fout.write("echo\n")
    fout.write("echo\n")
    fout.write("echo 'START---------------'\n")
    fout.write("start=$(date +%s)\n")
    fout.write("echo 'Work Directory ' ${PWD}\n")
    fout.write("cd "+str(path)+"\n")
    fout.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh\n")
    fout.write("python TrainEdgeClassificationFromGraph.py --config %s --do_validation --do_test\n" %(ConfigPath+ConfigFile))
    fout.write("end=$(date +%s)\n")
    fout.write('echo "Total time: $((end - start)) seconds"\n')
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
