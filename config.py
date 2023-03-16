feature_names = ['RC 보 단면 폭',
                'RC 보 단면 높이',
                '인장철근 유효깊이',
                '압축철근 유효깊이',
                'RC 보 길이',
                'RC 보 순지간',
                '하중재하길이',
                '콘크리트 28일 압축강도',
                '인장철근 단면적',
                '인장철근 항복강도',
                '압축철근 단면적',
                '압축철근 항복강도',
                'Carbon',
                'PBO',
                'Glass',
                'CFRP',
                'Basalt',
                '텍스타일 섬유 인장강도',
                '텍스타일 단면적',
                '텍스타일 레이어 수',
                '종방향 메쉬 크기',
                '횡방향 메쉬 크기',
                '전단보강 여부']

feature_names_label = ['빔 폭',
                '빔 높이',
                '인장철근 깊이',
                '압축철근 깊이',
                '철근콘크리트 길이',
                '빔 순지간',
                '하중재하길이',
                '콘크리트 28일 압축강도',
                '인장철근 단면적',
                '인장철근 항복강도',
                '압축철근 단면적',
                '압축철근 항복강도',
                'Carbon',
                'PBO',
                'Glass',
                'CFRP',
                'Basalt',
                '섬유 인장강도',
                '보강 텍스타일 단면적',
                '보강 텍스타일 레이어 수',
                '종방향 메쉬 크기',
                '횡방향 메쉬 크기',
                '전단보강 여부',
                '항복강도']


feature_list = ['b', 'h', 'd', "d'", 'L', 'l', 'a', 'fck', "As'", "fy'", 'As', 'fy',
                'C', 'PBO', 'G', 'CF', 'B', 'ff', 'Af', 'layer', 'Swr', 'Swf','Anc']

mean_dict={'b': 176.9625,
            'h': 242.3,
            'd': 207.9835,
            "d'": 25.825,
            'L': 2441.0625,
            'l': 2215.75,
            'a': 794.25,
            'fck': 30.697625,
            'As': 110.73037500000001,
            'fy': 357.99725,
            "As'": 230.72625,
            "fy'": 494.216,
            'C': 0.3625,
            'PBO': 0.1875,
            'G': 0.15,
            'CF': 0.075,
            'B': 0.0375,
            'ff': 3120.6375,
            'Af': 14.595875000000001,
            'layer': 2.125,
            'Swr': 9.932500000000001,
            'Swf': 13.569999999999999,
            'Anc': 0.325}

std_dict={'b': 96.34890291928602,
            'h': 81.40061424829668,
            'd': 72.65806763704909,
            "d'": 18.162994659471767,
            'L': 899.6255101950755,
            'l': 849.2920213330631,
            'a': 266.5298998236408,
            'fck': 8.828589814312078,
            'As': 107.1200526330592,
            'fy': 236.85287821754986,
            "As'": 194.7459760455592,
            "fy'": 93.60362070454326,
            'C': 0.4807221130757352,
            'PBO': 0.3903123748998999,
            'G': 0.3570714214271425,
            'CF': 0.26339134382131846,
            'B': 0.18998355191963331,
            'ff': 2076.1850245326764,
            'Af': 30.569788300450742,
            'layer': 2.0938899206978383,
            'Swr': 6.126331998022961,
            'Swf': 9.863523711128797,
            'Anc': 0.4683748498798799}
