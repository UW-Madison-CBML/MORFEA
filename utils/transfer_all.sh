#!/bin/bash
# build transfer_all.sub
FILES= "$(ls ../../../../staging/groups/bhaskar_group/"$1"/)"

while IFS="" read -r f || [ -n "$f" ]
do
  echo "
# transfer everything in staging to research drive
# transfer_all.sub
shell = echo \"moved\"
		
log = transfer_all_\$(Cluster)_\$(Process).log
error = transfer_all_\$(Cluster)_\$(Process).err
output = transfer_all_\$(Cluster)_\$(Process).out

transfer_input_files = file:///staging/groups/bhaskar_group/ivf/$f
transfer_output_files = $f
transfer_output_remaps = \"$f = pelican://chtc.wisc.edu/researchdrive/dbhaskar3/CHTC/ivf/$f\"
 
request_cpus = 1
request_memory = 8GB 
request_disk = 8GB
queue 1" > transfer_all.sub
condor_submit transfer_all.sub
done < $FILES

