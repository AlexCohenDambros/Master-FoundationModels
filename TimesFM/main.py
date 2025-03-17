from times_fm_code_forecasting_last_year import product_and_single_thread_testing, run_all_in_thread

if __name__ == "__main__":
    try:
        

        ''' 
        # INFO: SINGLE STATE (Testing model for a single state)
        ''' 
        product_and_single_thread_testing()

        ''' 
        # INFO: ALL STATES (Testing model for all states)s
        
        model: 200M or 500M
        type_prediction: 'zeroshot', 'fine_tuning_indiv', 'fine_tuning_global', 'fine_tuning_product'
        
        ''' 
        # run_all_in_thread(type_prediction='zeroshot', type_model='200M')
        # run_all_in_thread(type_prediction='zeroshot', type_model='500M')
        

    except Exception as e:
        print("An error occurred:", e)