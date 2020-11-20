#!/bin/bash
PERCENTAGE_PRIVILEGED="5 10 25 30 50"
N=30
EPOCHS=100



mkdir result_$(date +'%d_%m_%Y')
cd result_$(date +'%d_%m_%Y')
for i in ${PERCENTAGE_PRIVILEGED}
do
	mkdir privileged_$i
	cd privileged_$i
	for j in {1..30}
	do
		mkdir run_$j
		cd run_$j
		mkdir saved_models
		mkdir saved_images
		FILE_PRIV="../../../priv_$i/privileged_indices.txt"
		FILE_UNPRIV="../../../priv_$i/unprivileged_indices.txt"
		python ../../../architecture_input_updated.py $i $EPOCHS $FILE_PRIV $FILE_UNPRIV >> res.txt
		
		while IFS=" " read -r train_m1_acc train_m2_acc train_m3_acc train_m_total_acc train_m1_loss train_m2_loss train_m3_loss train_ce_final_loss train_total_loss val_m1_acc val_m2_acc val_m3_acc val_m_total_acc val_m1_loss val_m2_loss val_m3_loss val_m_total_loss sum_val_loss test_m1_acc test_m2_acc test_m3_acc test_total_acc test_m1_loss test_m2_loss test_m3_loss test_crossentropy_final test_total_loss precision_m1 precision_m2 precision_m3 precision_final recall_m1 recall_m2 recall_m3 recall_final fbeta_m1 fbeta_m2 fbeta_m3 fbeta_final 
		do
			echo " "
		done < res.txt
		
		echo "$train_m1_acc $train_m2_acc $train_m3_acc $train_m_total_acc $train_m1_loss $train_m2_loss $train_m3_loss $train_ce_final_loss $train_total_loss $val_m1_acc $val_m2_acc $val_m3_acc $val_m_total_acc $val_m1_loss $val_m2_loss $val_m3_loss $val_m_total_loss $sum_val_loss $test_m1_acc $test_m2_acc $test_m3_acc $test_total_acc $test_m1_loss $test_m2_loss $test_m3_loss $test_crossentropy_final $test_total_loss $precision_m1 $precision_m2 $precision_m3 $precision_final $recall_m1 $recall_m2 $recall_m3 $recall_final $fbeta_m1 $fbeta_m2 $fbeta_m3 $fbeta_final" >> "../res_$i.csv"
		
		cd ..
	done
	cd ../
done