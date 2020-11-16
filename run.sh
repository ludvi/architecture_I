#!/bin/bash
PERCENTAGE_PRIVILEGED="5 10 25 30 50"
N=100
cd data
for f in *.mat;
do
	mkdir ../result_$(date +'%d_%m_%Y')
	cd ../result_$(date +'%d_%m_%Y')
	mkdir ${f%.mat}
	for i in ${PERCENTAGE_PRIVILEGED}
	do
		mkdir ${f%.*}/privileged_$i
		cd ${f%.*}/privileged_$i
		for j in {1..100}
		do
			mkdir run_$j
			cd run_$j
			mkdir saved_models
			mkdir saved_images
			python ../../../../architecture_input_updated.py $f $i $j >> res.txt
			while IFS=" " read -r train_m1_acc train_m2_acc train_m3_acc train_m_total_acc train_m1_loss train_m2_loss train_m3_loss train_ce_final_loss train_total_loss train_m1_mse train_m2_mse train_m3_mse train_m_total_mse val_m1_acc val_m2_acc val_m3_acc val_m_total_acc val_m1_loss val_m2_loss val_m3_loss val_m_total_loss sum_val_loss val_m1_mse val_m2_mse val_m3_mse val_m_total_mse test_m1_acc test_m2_acc test_m3_acc test_total_acc test_m1_loss test_m2_loss test_m3_loss test_crossentropy_final test_total_loss test_m1_mse test_m2_mse test_m3_mse test_m_total_mse precision_m1 precision_m2 precision_m3 precision_final recall_m1 recall_m2 recall_m3 recall_final fbeta_m1 fbeta_m2 fbeta_m3 fbeta_final 
			do
				echo " "
			done < res.txt
			
			tr_m1_acc=$(awk "BEGIN {print $tr_m1_acc+$train_m1_acc}")
			tr_m2_acc=$(awk "BEGIN {print $tr_m2_acc+$train_m2_acc}")
			tr_m3_acc=$(awk "BEGIN {print $tr_m3_acc+$train_m3_acc}")
			tr_m_total_acc=$(awk "BEGIN {print $tr_m_total_acc+$train_m_total_acc}")
			tr_m1_loss=$(awk "BEGIN {print $tr_m1_loss+$train_m1_loss}")
			tr_m2_loss=$(awk "BEGIN {print $tr_m2_loss+$train_m2_loss}")
			tr_m3_loss=$(awk "BEGIN {print $tr_m3_loss+$train_m3_loss}")
			tr_ce_final_loss=$(awk "BEGIN {print $tr_ce_final_loss+$train_ce_final_loss}")
			tr_total_loss=$(awk "BEGIN {print $tr_total_loss+$train_total_loss}")
			tr_m1_mse=$(awk "BEGIN {print $tr_m1_mse+$train_m1_mse}")
			tr_m2_mse=$(awk "BEGIN {print $tr_m2_mse+$train_m2_mse}")
			tr_m3_mse=$(awk "BEGIN {print $tr_m3_mse+$train_m3_mse}")
			tr_m_total_mse=$(awk "BEGIN {print $tr_m_total_mse+$train_m_total_mse}")
			v_m1_acc=$(awk "BEGIN {print $v_m1_acc+$val_m1_acc}")
			v_m2_acc=$(awk "BEGIN {print $v_m2_acc+$val_m2_acc}")
			v_m3_acc=$(awk "BEGIN {print $v_m3_acc+$val_m3_acc}")
			v_m_total_acc=$(awk "BEGIN {print $v_m_total_acc+$val_m_total_acc}")
			v_m1_loss=$(awk "BEGIN {print $v_m1_loss+$val_m1_loss}")
			v_m2_loss=$(awk "BEGIN {print $v_m2_loss+$val_m2_loss}")
			v_m3_loss=$(awk "BEGIN {print $v_m3_loss+$val_m3_loss}")
			v_ce_final_loss=$(awk "BEGIN {print $v_ce_final_loss+$val_m_total_loss}")
			v_total_loss=$(awk "BEGIN {print $v_total_loss+$sum_val_loss}")
			v_m1_mse=$(awk "BEGIN {print $v_m1_mse+$val_m1_mse}")
			v_m2_mse=$(awk "BEGIN {print $v_m2_mse+$val_m2_mse}")
			v_m3_mse=$(awk "BEGIN {print $v_m3_mse+$val_m3_mse}")
			v_m_total_mse=$(awk "BEGIN {print $v_m_total_mse+$val_m_total_mse}")
			te_m1_acc=$(awk "BEGIN {print $te_m1_acc+$test_m1_acc}")
			te_m2_acc=$(awk "BEGIN {print $te_m2_acc+$test_m2_acc}")
			te_m3_acc=$(awk "BEGIN {print $te_m3_acc+$test_m3_acc}")
			te_total_acc=$(awk "BEGIN {print $te_total_acc+$test_total_acc}")
			te_m1_loss=$(awk "BEGIN {print $te_m1_loss+$test_m1_loss}")
			te_m2_loss=$(awk "BEGIN {print $te_m2_loss+$test_m2_loss}")
			te_m3_loss=$(awk "BEGIN {print $te_m3_loss+$test_m3_loss}")			
			ce_final=$(awk "BEGIN {print $ce_final+$test_crossentropy_final}")
			te_total_loss=$(awk "BEGIN {print $te_total_loss+$test_total_loss}")
			te_m1_mse=$(awk "BEGIN {print $te_m1_mse+$test_m1_mse}")
			te_m2_mse=$(awk "BEGIN {print $te_m2_mse+$test_m2_mse}")
			te_m3_mse=$(awk "BEGIN {print $te_m3_mse+$test_m3_mse}")
			te_m_total_mse=$(awk "BEGIN {print $te_m_total_mse+$test_m_total_mse}")
			pre_m1=$(awk "BEGIN {print $pre_m1+$precision_m1}")
			pre_m2=$(awk "BEGIN {print $pre_m2+$precision_m2}")
			pre_m3=$(awk "BEGIN {print $pre_m3+$precision_m3}")
			pre_final=$(awk "BEGIN {print $pre_final+$precision_final}")
			re_m1=$(awk "BEGIN {print $re_m1+$recall_m1}")
			re_m2=$(awk "BEGIN {print $re_m2+$recall_m2}")
			re_m3=$(awk "BEGIN {print $re_m3+$recall_m3}")
			re_final=$(awk "BEGIN {print $re_final+$recall_final}")
			fb_m1=$(awk "BEGIN {print $fb_m1+$fbeta_m1}")
			fb_m2=$(awk "BEGIN {print $fb_m2+$fbeta_m2}")
			fb_m3=$(awk "BEGIN {print $fb_m3+$fbeta_m3}")
			fb_final=$(awk "BEGIN {print $fb_final+$fbeta_final}")
			
			cd ..
		done
		
		tr_m1_acc=$(awk "BEGIN {print $tr_m1_acc/$N}")
		tr_m2_acc=$(awk "BEGIN {print $tr_m2_acc/$N}")
		tr_m3_acc=$(awk "BEGIN {print $tr_m3_acc/$N}")
		tr_m_total_acc=$(awk "BEGIN {print $tr_m_total_acc/$N}")
		tr_m1_loss=$(awk "BEGIN {print $tr_m1_loss/$N}")
		tr_m2_loss=$(awk "BEGIN {print $tr_m2_loss/$N}")
		tr_m3_loss=$(awk "BEGIN {print $tr_m3_loss/$N}")
		tr_ce_final_loss=$(awk "BEGIN {print $tr_ce_final_loss/$N}")
		tr_total_loss=$(awk "BEGIN {print $tr_total_loss/$N}")
		tr_m1_mse=$(awk "BEGIN {print $tr_m1_mse/$N}")
		tr_m2_mse=$(awk "BEGIN {print $tr_m2_mse/$N}")
		tr_m3_mse=$(awk "BEGIN {print $tr_m3_mse/$N}")
		tr_m_total_mse=$(awk "BEGIN {print $tr_m_total_mse/$N}")
		v_m1_acc=$(awk "BEGIN {print $v_m1_acc/$N}")
		v_m2_acc=$(awk "BEGIN {print $v_m2_acc/$N}")
		v_m3_acc=$(awk "BEGIN {print $v_m3_acc/$N}")
		v_m_total_acc=$(awk "BEGIN {print $v_m_total_acc/$N}")
		v_m1_loss=$(awk "BEGIN {print $v_m1_loss/$N}")
		v_m2_loss=$(awk "BEGIN {print $v_m2_loss/$N}")
		v_m3_loss=$(awk "BEGIN {print $v_m3_loss/$N}")
		v_ce_final_loss=$(awk "BEGIN {print $v_ce_final_loss/$N}")
		v_total_loss=$(awk "BEGIN {print $v_total_loss/$N}")
		v_m1_mse=$(awk "BEGIN {print $v_m1_mse/$N}")
		v_m2_mse=$(awk "BEGIN {print $v_m2_mse/$N}")
		v_m3_mse=$(awk "BEGIN {print $v_m3_mse/$N}")
		v_m_total_mse=$(awk "BEGIN {print $v_m_total_mse/$N}")
		te_m1_acc=$(awk "BEGIN {print $te_m1_acc/$N}")
		te_m2_acc=$(awk "BEGIN {print $te_m2_acc/$N}")
		te_m3_acc=$(awk "BEGIN {print $te_m3_acc/$N}")
		te_total_acc=$(awk "BEGIN {print $te_total_acc/$N}")
		te_m1_loss=$(awk "BEGIN {print $te_m1_loss/$N}")
		te_m2_loss=$(awk "BEGIN {print $te_m2_loss/$N}")
		te_m3_loss=$(awk "BEGIN {print $te_m3_loss/$N}")
		ce_final=$(awk "BEGIN {print $ce_final/$N}")
		te_total_loss=$(awk "BEGIN {print $te_total_loss/$N}")
		te_m1_mse=$(awk "BEGIN {print $te_m1_mse/$N}")
		te_m2_mse=$(awk "BEGIN {print $te_m2_mse/$N}")
		te_m3_mse=$(awk "BEGIN {print $te_m3_mse/$N}")
		te_m_total_mse=$(awk "BEGIN {print $te_m_total_mse/$N}")
		pre_m1=$(awk "BEGIN {print $pre_m1/$N}")
		pre_m2=$(awk "BEGIN {print $pre_m2/$N}")
		pre_m3=$(awk "BEGIN {print $pre_m3/$N}")
		pre_final=$(awk "BEGIN {print $pre_final/$N}")
		re_m1=$(awk "BEGIN {print $re_m1/$N}")
		re_m2=$(awk "BEGIN {print $re_m2/$N}")
		re_m3=$(awk "BEGIN {print $re_m3/$N}")
		re_final=$(awk "BEGIN {print $re_final/$N}")
		fb_m1=$(awk "BEGIN {print $fb_m1/$N}")
		fb_m2=$(awk "BEGIN {print $fb_m2/$N}")
		fb_m3=$(awk "BEGIN {print $fb_m3/$N}")
		fb_final=$(awk "BEGIN {print $fb_final/$N}")
		
		echo "priv_$i $tr_m1_acc $tr_m2_acc $tr_m3_acc  $tr_m_total_acc $tr_m1_loss $tr_m2_loss $tr_m3_loss $tr_ce_final_loss $tr_total_loss $tr_m1_mse $tr_m2_mse $tr_m3_mse $tr_m_total_mse $v_m1_acc $v_m2_acc $v_m3_acc $v_m_total_acc $v_m1_loss $v_m2_loss $v_m3_loss $v_ce_final_loss $v_total_loss $v_m1_mse $v_m2_mse $v_m3_mse $v_m_total_mse $te_m1_acc $te_m2_acc $te_m3_acc $te_total_acc $te_m1_loss $te_m2_loss $te_m3_loss $ce_final $te_total_loss $te_m1_mse $te_m2_mse $te_m3_mse $te_m_total_mse $pre_m1 $pre_m2 $pre_m3 $pre_final $re_m1 $re_m2 $re_m3 $re_final $fb_m1 $fb_m2 $fb_m3 $fb_final" >> "../final_result_${f%.*}.csv"
		
		tr_m1_acc=0
		tr_m2_acc=0
		tr_m3_acc=0
		tr_m_total_acc=0
		tr_m1_loss=0
		tr_m2_loss=0
		tr_m3_loss=0
		tr_ce_final_loss=0
		tr_total_loss=0
		tr_m1_mse=0
		tr_m2_mse=0
		tr_m3_mse=0
		tr_m_total_mse=0
		v_m1_acc=0
		v_m2_acc=0
		v_m3_acc=0
		v_m_total_acc=0
		v_m1_loss=0
		v_m2_loss=0
		v_m3_loss=0
		v_ce_final_loss=0
		v_total_loss=0
		v_m1_mse=0
		v_m2_mse=0
		v_m3_mse=0
		v_m_total_mse=0
		te_m1_acc=0
		te_m2_acc=0
		te_m3_acc=0
		te_total_acc=0
		te_m1_loss=0
		te_m2_loss=0
		te_m3_loss=0
		ce_final=0
		te_total_loss=0
		te_m1_mse=0
		te_m2_mse=0
		te_m3_mse=0
		te_m_total_mse=0
		pre_m1=0
		pre_m2=0
		pre_m3=0
		pre_final=0
		re_m1=0
		re_m2=0
		re_m3=0
		re_final=0
		fb_m1=0
		fb_m2=0
		fb_m3=0
		fb_final=0
		
		cd ../../
	done
done




