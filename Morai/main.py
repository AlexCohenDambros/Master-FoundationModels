###  Code Description

'''

NOTE: Uncomment and comment out the code below as needed.

'''

from morai_code_forecasting import product_and_single_thread_testing, run_all_in_thread

if __name__ == "__main__":
    try:
        

        ''' 
        # INFO: SINGLE STATE (Testing model for a single state)
        ''' 
        # product_and_single_thread_testing()

        ''' 
        # INFO: ALL STATES small/base (Testing model for all states)
        
        model: small, base, large
        
        ''' 
        run_all_in_thread(type_prediction='zeroshot', type_model='small')
        run_all_in_thread(type_prediction='zeroshot', type_model='base')
        run_all_in_thread(type_prediction='zeroshot', type_model='large')

    except Exception as e:
        print("An error occurred:", e)