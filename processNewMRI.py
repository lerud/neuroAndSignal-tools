# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: neuroTools
#     language: python
#     name: neurotools
# ---

# %%
import os
import glob
import mne

# %%
parentDir='/Users/karl/Dropbox/UMD/'

subjectDir='R3093/'

# %%
# Make sure the entire MRI dataset is converted to NIfTI and inside a directory called mri inside the subject directory, then that's all that needs to be done to prepare for this
# as long as there are exactly 2 T1s and we want the second one
allT1s=sorted(glob.glob(parentDir+subjectDir+'mri/'+'_T1*nii'))
T1toUse=allT1s[1]

print(f'''Found the following T1s: {allT1s}

Using {T1toUse}''')
os.system('fslinfo '+T1toUse)

# %%
os.system('mkdir '+parentDir+subjectDir+'mri/simnibsParc')
os.system('mkdir '+parentDir+subjectDir+'mri/simnibsParc/org')
os.system('cp '+T1toUse+' '+parentDir+subjectDir+'mri/simnibsParc/org/T1.nii')

# %%
#os.chdir(parentDir+subjectDir+'mri/simnibsParc')
#os.system('charm '+subjectDir[:-1]+' org/T1.nii --forceqform')
#os.chdir(parentDir)

# %%
os.system('recon-all -s '+subjectDir[:-1]+' -i '+subjectDir+'mri/simnibsParc/org/T1.nii -all')

# %%
mne.bem.make_watershed_bem(subjectDir[:-1])

# %%

# %%
