#!/bin/bash
rm export_latents_scripts/*
cat <<EOF > "export_latents_scripts/latents.dag"
JOB  0 export_latents.sub
PARENT 0 CHILD 1
EOF

FILE="get_latents.txt"
index=1
last_index=$(wc -l < "$FILE")
while IFS= read -r line; do
	cat <<EOF > "export_latents_scripts/${line}_latents.sub"
   
shell = ./${line}_latents.sh
   		
log = export_latents_scripts/${line}_latents_\$(Cluster)_\$(Process).log
error = export_latents_scripts/${line}_latents_\$(Cluster)_\$(Process).err
output = export_latents_scripts/${line}_latents_\$(Cluster)_\$(Process).out
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
Requirements = (Target.HasCHTCStaging == true)
# Transfer our executable script
transfer_output_remaps = "latents.tar.gz =  file:///staging/groups/bhaskar_group/ivf/latents.tar.gz;"
transfer_input_files = export_latents_scripts/${line}_latents.sh, file:///staging/groups/bhaskar_group/ivf/${line}_latents.tar.gz, file:///staging/groups/bhaskar_group/ivf/latents.tar.gz
#Requirements = (Machine == "bhaskargpu4000.chtc.wisc.edu") && (DriverVersion == "12.8") && (DeviceName == "NVIDIA H200")
 
transfer_output_files = latents.tar.gz
# Requirements (e.g., operating system) your job needs, what amount of
# compute resources each job will need on the computer where it runs.
request_cpus = 1
request_memory = 32GB
request_disk = 32GB


queue 1
EOF

cat <<EOF > "export_latents_scripts/${line}_latents.sh"
#!/bin/bash
tar -xzvf ${line}_latents.tar.gz

tar -xzvf latents.tar.gz

mv ${line}_latents/* latents

tar -czvf latents.tar.gz latents/
EOF


cat <<EOF >> "export_latents_scripts/latents.dag"
JOB  $index export_latents_scripts/${line}_latents.sub
EOF
if [[ "$index" -ne "$last_index" ]]; then
cat <<EOF >> "export_latents_scripts/latents.dag"
PARENT $index CHILD $((index + 1))
EOF
fi
	((index++))

done < "$FILE"

chmod +x *.sh
condor_submit_dag export_latents_scripts/latents.dag


