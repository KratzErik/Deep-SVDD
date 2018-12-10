def files_equal(file1,file2, start_str=None):
    with open(file1) as f1, open(file2) as f2:
        lines1 = list(filter(None, (line.rstrip() for line in f1)))
        lines2 = list(filter(None, (line.rstrip() for line in f2)))

    if len(lines1) != len(lines2):
        return False
    else:
        line_count = 0
        if start_str is not None:
            found_start_str = False
            while not found_start_str:
                found_start_str = (start_str in lines1[line_count].strip() or start_str in lines2[line_count].strip())
                line_count += 1
        while line_count < len(lines1):
            if lines1[line_count].strip() != lines2[line_count].strip():
                return False
            line_count += 1
    return True
