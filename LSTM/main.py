from long_short_term_memory_pytorch import product_and_single_thread_testing, run_lstm_in_thread

if __name__ == "__main__":
    try:

        ''' 
        # INFO: TEST PYTORCH
        ''' 
        
        # product_and_single_thread_testing()
        
        ''' 
        # INFO: ALL STATES RECURSIVE/DIRECT PYTORCH
        ''' 
        run_lstm_in_thread("recursive")


    except Exception as e:
        print("An error occurred:", e)