from model_time_moe import derivative_and_single_thread_testing, run_all_in_thread
from forecasting_cities import product_and_single_thread_testing_5_years_for_cities, run_all_in_thread_5_years_for_cities

if __name__ == "__main__":
    try:

        ''' 
        # INFO: SINGLE STATE (Testing model for a single state)
        ''' 
        
        # derivative_and_single_thread_testing()

        ''' 
        # INFO: ALL STATES TimeMoE-50M/TimeMoE-200M (Testing model for all states)
        
        model: 50M or 200M
        type_prediction: 'zeroshot', 'fine_tuning_indiv', 'fine_tuning_global', 'fine_tuning_product'

        ''' 
        run_all_in_thread(type_prediction='fine_tuning_product', type_model='50M')
        run_all_in_thread(type_prediction='fine_tuning_product', type_model='200M')
    
    
        ''' 
        # INFO: SINGLE STATE 5 YEARS FOR CITIES (Testing model for a single state over 5 years)
        ''' 
        
        # product_and_single_thread_testing_5_years_for_cities()


        ''' 
        # INFO: ALL STATES TimeMoE-50M/TimeMoE-200M 5 YEARS FOR CITIES (Testing model for all states over 5 years)
        
        'TimeMoE-50M_ZERO_SHOT'
        'TimeMoE-200M_ZERO_SHOT'

        '''
        # run_all_in_thread_5_years_for_cities(type_model='TimeMoE-50M_ZERO_SHOT')
        # run_all_in_thread_5_years_for_cities(type_model='TimeMoE-200M_ZERO_SHOT')


    except Exception as e:
        print("An error occurred:", e)