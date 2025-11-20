date +"%T"
condor_submit train_model.sub
condor_watch_q
