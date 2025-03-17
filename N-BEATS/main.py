from n_beats import run_nbeats_in_thread


if __name__ == "__main__":
    try:

        ''' 
        # INFO: ALL STATES N-BEATS 
        ''' 

        run_nbeats_in_thread()

    except Exception as e:
        print("An error occurred:", e)