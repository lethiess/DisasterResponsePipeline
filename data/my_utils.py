def getFileName(filepath):
    '''
    Extract the file name from the file path. This includes
    removing the path and the file extension.

    Args:
        database_filepath: The path to the file 

    Return:
        Name of the Database without path and file extension.
    '''
    file_name = ""
    # split path from file name
    try:
        file_name = filepath.rsplit("\\",1)[1]
    except:
        file_name = filepath
    # split file extension and return name 
    return file_name.split(".")[0]