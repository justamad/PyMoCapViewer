vicon_skeleton = [
    # Right lower body
    ["RTOE", "RHEE"],
    ["RHEE", "RANK"],
    ["RANK", "RTOE"],

    ["RANK", "RTIB"],
    ["RANK", "RKNE"],
    ["RTIB", "RKNE"],
    ["RKNE", "RASI"],
    ["RKNE", "RTHI"],
    ["RTHI", "RASI"],
    ["RTHI", "RPSI"],

    # Left lower body
    ["LTOE", "LHEE"],
    ["LHEE", "LANK"],
    ["LANK", "LTOE"],

    ["LANK", "LTIB"],
    ["LANK", "LKNE"],
    ["LTIB", "LKNE"],
    ["LKNE", "LASI"],
    ["LKNE", "LTHI"],
    ["LTHI", "LASI"],
    ["LTHI", "LPSI"],

    # Right upper body
    ["RFIN", "RWRA"],
    ["RFIN", "RWRB"],
    ["RWRA", "RWRB"],
    ["RWRA", "RFRM"],
    ["RWRB", "RFRM"],
    ["RFRM", "RELB"],
    ["RELB", "RUPA"],
    ["RUPA", "RSHO"],
    ["RSHO", "CLAV"],

    # Left upper body
    ["LFIN", "LWRA"],
    ["LFIN", "LWRB"],
    ["LWRA", "LWRB"],
    ["LWRA", "LFRM"],
    ["LWRB", "LFRM"],
    ["LFRM", "LELB"],
    ["LELB", "LUPA"],
    ["LUPA", "LSHO"],
    ["LSHO", "CLAV"],

    # Center upper body
    ["T10", "RBAK"],
    ["RBAK", "C7"],
    ["C7", "T10"],
    ["CLAV", "STRN"],
    ["T10", "STRN"],
    ["CLAV", "C7"],
    ["STRN", "RBAK"],
    ["CLAV", "RBAK"],
    ["T10", "CLAV"],

    # Hip
    ["RASI", "LASI"],
    ["LASI", "LPSI"],
    ["LPSI", "RPSI"],
    ["RPSI", "RASI"],
    ["RASI", "LPSI"],
    ["LASI", "RPSI"],

    # Head
    ["RFHD", "LFHD"],
    ["LFHD", "LBHD"],
    ["LBHD", "RBHD"],
    ["RBHD", "RFHD"],
    ["RFHD", "LBHD"],
    ["LFHD", "RBHD"],

]
