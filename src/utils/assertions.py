def files_equal(file1,file2):
    with open(file1) as f1, open(file2) as f2: 
        lines1 = list(filter(None, (line.rstrip() for line in f1)))
        lines2 = list(filter(None, (line.rstrip() for line in f2)))

    if len(lines1) != len(lines2):
        return False
    else:
        line_count = 0
        while line_count < len(lines1):
            if lines1[line_count].strip() != lines2[line_count].strip():
                return False   
            line_count += 1     
    return True