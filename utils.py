def chunks(lst, size):
    """Yield successive n-sized chunks from list"""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]
