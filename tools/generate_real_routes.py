"""
真实航线数据生成器
基于实际航空网络结构生成300条全球代表性航线
每条航线包含2-10个真实控制点
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import csv


class RouteGenerator:
    """航线生成器 - 基于真实航空网络"""
    
    def __init__(self):
        # 全球主要机场数据库（IATA代码、纬度、经度、城市名、国家）
        self.airports = {
            # 亚洲 - 中国
            'PEK': (40.0799, 116.6031, '北京', '中国'),
            'PVG': (31.1443, 121.8083, '上海', '中国'),
            'CAN': (23.3924, 113.2988, '广州', '中国'),
            'CTU': (30.5785, 103.9470, '成都', '中国'),
            'SZX': (22.6393, 113.8107, '深圳', '中国'),
            'HGH': (30.2295, 120.4347, '杭州', '中国'),
            'XIY': (34.4471, 108.7515, '西安', '中国'),
            'KMG': (25.1019, 102.9292, '昆明', '中国'),
            'CKG': (29.7192, 106.6417, '重庆', '中国'),
            'NKG': (31.7420, 118.8620, '南京', '中国'),
            'WUH': (30.7838, 114.2081, '武汉', '中国'),
            'XMN': (24.5440, 118.1278, '厦门', '中国'),
            'TSN': (39.1244, 117.3464, '天津', '中国'),
            'CGO': (34.5197, 113.8408, '郑州', '中国'),
            'HRB': (45.6234, 126.2500, '哈尔滨', '中国'),
            'URC': (43.9071, 87.4744, '乌鲁木齐', '中国'),
            'DLC': (38.9657, 121.5386, '大连', '中国'),
            'SYX': (18.3029, 109.4125, '三亚', '中国'),
            'TAO': (36.2661, 120.3742, '青岛', '中国'),
            'CSX': (28.1892, 113.2200, '长沙', '中国'),
            'NNG': (22.6083, 108.1720, '南宁', '中国'),
            'FOC': (25.9351, 119.6633, '福州', '中国'),
            'HFE': (31.7800, 117.2980, '合肥', '中国'),
            'SJW': (38.2807, 114.6964, '石家庄', '中国'),
            
            # 亚洲 - 其他国家
            'HKG': (22.3080, 113.9185, '香港', '中国'),
            'TPE': (25.0777, 121.2328, '台北', '中国'),
            'NRT': (35.7647, 140.3864, '东京成田', '日本'),
            'HND': (35.5494, 139.7798, '东京羽田', '日本'),
            'KIX': (34.4273, 135.2440, '大阪', '日本'),
            'NGO': (35.2552, 136.8049, '名古屋', '日本'),
            'FUK': (33.5859, 130.4510, '福冈', '日本'),
            'ICN': (37.4602, 126.4407, '首尔', '韩国'),
            'GMP': (37.5583, 126.7906, '首尔金浦', '韩国'),
            'PUS': (35.1795, 129.0756, '釜山', '韩国'),
            'SIN': (1.3644, 103.9915, '新加坡', '新加坡'),
            'BKK': (13.6900, 100.7501, '曼谷', '泰国'),
            'CNX': (18.7668, 98.9628, '清迈', '泰国'),
            'HKT': (8.1132, 98.3169, '普吉岛', '泰国'),
            'KUL': (2.7456, 101.7099, '吉隆坡', '马来西亚'),
            'MNL': (14.5086, 121.0194, '马尼拉', '菲律宾'),
            'CGK': (6.1256, 106.6559, '雅加达', '印尼'),
            'DPS': (8.7482, 115.1672, '巴厘岛', '印尼'),
            'HAN': (21.2212, 105.8072, '河内', '越南'),
            'SGN': (10.8188, 106.6520, '胡志明市', '越南'),
            'DAD': (16.0439, 108.1994, '岘港', '越南'),
            'DEL': (28.5562, 77.1000, '德里', '印度'),
            'BOM': (19.0896, 72.8656, '孟买', '印度'),
            'BLR': (13.1986, 77.7066, '班加罗尔', '印度'),
            'MAA': (12.9941, 80.1709, '金奈', '印度'),
            'ISB': (33.6169, 73.0992, '伊斯兰堡', '巴基斯坦'),
            'DXB': (25.2532, 55.3657, '迪拜', '阿联酋'),
            'AUH': (24.4330, 54.6511, '阿布扎比', '阿联酋'),
            'DOH': (25.2731, 51.6080, '多哈', '卡塔尔'),
            'RUH': (24.9574, 46.6987, '利雅得', '沙特'),
            'JED': (21.6796, 39.1565, '吉达', '沙特'),
            'TLV': (32.0114, 34.8867, '特拉维夫', '以色列'),
            
            # 北美
            'JFK': (40.6413, -73.7781, '纽约肯尼迪', '美国'),
            'EWR': (40.6895, -74.1745, '纽约纽瓦克', '美国'),
            'LGA': (40.7769, -73.8740, '纽约拉瓜迪亚', '美国'),
            'LAX': (33.9416, -118.4085, '洛杉矶', '美国'),
            'ORD': (41.9742, -87.9073, '芝加哥', '美国'),
            'DFW': (32.8998, -97.0403, '达拉斯', '美国'),
            'SFO': (37.6213, -122.3790, '旧金山', '美国'),
            'MIA': (25.7959, -80.2870, '迈阿密', '美国'),
            'ATL': (33.6407, -84.4277, '亚特兰大', '美国'),
            'BOS': (42.3656, -71.0096, '波士顿', '美国'),
            'SEA': (47.4502, -122.3088, '西雅图', '美国'),
            'LAS': (36.0840, -115.1537, '拉斯维加斯', '美国'),
            'PHX': (33.4352, -112.0101, '凤凰城', '美国'),
            'IAH': (29.9902, -104.8420, '休斯顿', '美国'),
            'DEN': (39.8561, -104.6737, '丹佛', '美国'),
            'MCO': (28.4312, -81.3081, '奥兰多', '美国'),
            'DTW': (42.2162, -83.3554, '底特律', '美国'),
            'MSP': (44.8848, -93.2223, '明尼阿波利斯', '美国'),
            'CLT': (35.2144, -80.9473, '夏洛特', '美国'),
            'PDX': (45.5898, -122.5951, '波特兰', '美国'),
            'SAN': (32.7336, -117.1897, '圣地亚哥', '美国'),
            'YVR': (49.1967, -123.1815, '温哥华', '加拿大'),
            'YYZ': (43.6777, -79.6248, '多伦多', '加拿大'),
            'YUL': (45.4657, -73.7455, '蒙特利尔', '加拿大'),
            'YYC': (51.1315, -114.0106, '卡尔加里', '加拿大'),
            'MEX': (19.4363, -99.0721, '墨西哥城', '墨西哥'),
            
            # 欧洲
            'LHR': (51.4700, -0.4543, '伦敦希思罗', '英国'),
            'LGW': (51.1537, -0.1821, '伦敦盖特威克', '英国'),
            'CDG': (49.0097, 2.5479, '巴黎戴高乐', '法国'),
            'ORY': (48.7262, 2.3659, '巴黎奥利', '法国'),
            'FRA': (50.0379, 8.5622, '法兰克福', '德国'),
            'MUC': (48.3537, 11.7750, '慕尼黑', '德国'),
            'AMS': (52.3105, 4.7683, '阿姆斯特丹', '荷兰'),
            'MAD': (40.4719, -3.5626, '马德里', '西班牙'),
            'BCN': (41.2974, 2.0833, '巴塞罗那', '西班牙'),
            'FCO': (41.8003, 12.2389, '罗马', '意大利'),
            'MXP': (45.6301, 8.7231, '米兰', '意大利'),
            'VCE': (45.5053, 12.3519, '威尼斯', '意大利'),
            'ZRH': (47.4582, 8.5481, '苏黎世', '瑞士'),
            'VIE': (48.1103, 16.5697, '维也纳', '奥地利'),
            'BRU': (50.9010, 4.4856, '布鲁塞尔', '比利时'),
            'CPH': (55.6180, 12.6506, '哥本哈根', '丹麦'),
            'ARN': (59.6519, 17.9186, '斯德哥尔摩', '瑞典'),
            'OSL': (60.1939, 11.1004, '奥斯陆', '挪威'),
            'HEL': (60.3172, 24.9633, '赫尔辛基', '芬兰'),
            'WAW': (52.1657, 20.9671, '华沙', '波兰'),
            'PRG': (50.1008, 14.2600, '布拉格', '捷克'),
            'BUD': (47.4360, 19.2556, '布达佩斯', '匈牙利'),
            'ATH': (37.9364, 23.9445, '雅典', '希腊'),
            'IST': (41.2753, 28.7519, '伊斯坦布尔', '土耳其'),
            'SAW': (40.8986, 29.3092, '伊斯坦布尔萨比哈', '土耳其'),
            'SVO': (55.9726, 37.4146, '莫斯科', '俄罗斯'),
            'LED': (59.8003, 30.2625, '圣彼得堡', '俄罗斯'),
            
            # 大洋洲
            'SYD': (-33.9461, 151.1772, '悉尼', '澳大利亚'),
            'MEL': (-37.6690, 144.8410, '墨尔本', '澳大利亚'),
            'BNE': (-27.3942, 153.1218, '布里斯班', '澳大利亚'),
            'PER': (-31.9403, 115.9672, '珀斯', '澳大利亚'),
            'AKL': (-37.0082, 174.7850, '奥克兰', '新西兰'),
            'CHC': (-43.4894, 172.5320, '基督城', '新西兰'),
            
            # 南美
            'GRU': (-23.4356, -46.4731, '圣保罗', '巴西'),
            'GIG': (-22.8099, -43.2505, '里约热内卢', '巴西'),
            'BSB': (-15.8697, -47.9208, '巴西利亚', '巴西'),
            'EZE': (-34.8222, -58.5358, '布宜诺斯艾利斯', '阿根廷'),
            'SCL': (-33.3930, -70.7858, '圣地亚哥', '智利'),
            'LIM': (-12.0219, -77.1143, '利马', '秘鲁'),
            'BOG': (4.7016, -74.1469, '波哥大', '哥伦比亚'),
            
            # 非洲
            'CAI': (30.1219, 31.4056, '开罗', '埃及'),
            'JNB': (-26.1392, 28.2460, '约翰内斯堡', '南非'),
            'CPT': (-33.9715, 18.6021, '开普敦', '南非'),
            'NBO': (-1.3192, 36.9278, '内罗毕', '肯尼亚'),
            'ADD': (8.9779, 38.7993, '亚的斯亚贝巴', '埃塞俄比亚'),
            'LOS': (6.5774, 3.3212, '拉各斯', '尼日利亚'),
            'CMN': (33.3676, -7.5898, '卡萨布兰卡', '摩洛哥'),
        }
        
    def calculate_waypoints_realistic(self, lat1: float, lon1: float, lat2: float, lon2: float, 
                                     num_waypoints: int) -> List[Tuple[float, float]]:
        """
        基于大圆路径计算真实航路点，模拟实际航路结构
        考虑地球曲率和实际航空导航习惯
        """
        # 转换为弧度
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # 计算大圆距离
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        if c < 0.001:  # 非常近的距离
            return []
        
        # 生成航路点
        waypoints = []
        
        if num_waypoints <= 0:
            return waypoints
        
        # 改进的分布策略：避免中间大段空白
        if num_waypoints <= 4:
            # 对于少量航路点，采用均匀分布，覆盖0.2到0.8的范围
            fractions = np.linspace(0.2, 0.8, num_waypoints).tolist()
        else:
            # 对于较多航路点，两端稍微密集一些，但中间也要保证有覆盖
            # 起点附近 (20%的点)
            start_points = int(np.ceil(num_waypoints * 0.2))
            # 终点附近 (20%的点)
            end_points = int(np.ceil(num_waypoints * 0.2))
            # 中间 (60%的点)
            middle_points = num_waypoints - start_points - end_points
            
            fractions = []
            # 起点段（0.1-0.25）
            if start_points > 0:
                fractions.extend(np.linspace(0.1, 0.25, start_points).tolist())
            # 中间段（0.3-0.7）- 确保覆盖中间
            if middle_points > 0:
                fractions.extend(np.linspace(0.3, 0.7, middle_points).tolist())
            # 终点段（0.75-0.9）
            if end_points > 0:
                fractions.extend(np.linspace(0.75, 0.9, end_points).tolist())
        
        for f in fractions:
            # 球面插值（Slerp）
            A = np.sin((1 - f) * c) / np.sin(c)
            B = np.sin(f * c) / np.sin(c)
            
            x = A * np.cos(lat1_rad) * np.cos(lon1_rad) + B * np.cos(lat2_rad) * np.cos(lon2_rad)
            y = A * np.cos(lat1_rad) * np.sin(lon1_rad) + B * np.cos(lat2_rad) * np.sin(lon2_rad)
            z = A * np.sin(lat1_rad) + B * np.sin(lat2_rad)
            
            lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
            lon = np.degrees(np.arctan2(y, x))
            
            # 添加小量随机偏移模拟实际航路的不规则性（±0.5度以内）
            lat += np.random.uniform(-0.3, 0.3)
            lon += np.random.uniform(-0.3, 0.3)
            
            waypoints.append((round(lat, 4), round(lon, 4)))
        
        return waypoints
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算两点间的大圆距离（公里）"""
        R = 6371.0
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def determine_waypoint_count(self, distance_km: float) -> int:
        """根据距离确定航路点数量"""
        if distance_km < 500:
            return np.random.randint(3, 5)  # 短程：3-4个
        elif distance_km < 1500:
            return np.random.randint(4, 7)  # 中短程：4-6个
        elif distance_km < 3000:
            return np.random.randint(5, 8)  # 中程：5-7个
        elif distance_km < 5000:
            return np.random.randint(5, 7)  # 中长程：5-6个
        elif distance_km < 8000:
            return np.random.randint(6, 9)  # 长程：6-8个
        else:
            return np.random.randint(8, 11)  # 超长程：8-10个
    
    def generate_routes(self) -> List[dict]:
        """生成300条全球代表性航线"""
        routes = []
        
        # 1. 中国国内主要航线（80条）
        china_domestic_pairs = [
            # 一线城市互飞
            ('PEK', 'PVG'), ('PEK', 'CAN'), ('PEK', 'CTU'), ('PEK', 'SZX'),
            ('PVG', 'CAN'), ('PVG', 'CTU'), ('PVG', 'SZX'), ('PVG', 'CKG'),
            ('CAN', 'CTU'), ('CAN', 'CKG'), ('CAN', 'HGH'), ('CAN', 'XIY'),
            # 北京出发
            ('PEK', 'HGH'), ('PEK', 'XIY'), ('PEK', 'KMG'), ('PEK', 'CKG'),
            ('PEK', 'NKG'), ('PEK', 'WUH'), ('PEK', 'XMN'), ('PEK', 'TSN'),
            ('PEK', 'CGO'), ('PEK', 'HRB'), ('PEK', 'URC'), ('PEK', 'DLC'),
            ('PEK', 'SYX'), ('PEK', 'TAO'), ('PEK', 'CSX'), ('PEK', 'NNG'),
            # 上海出发
            ('PVG', 'HGH'), ('PVG', 'XIY'), ('PVG', 'KMG'), ('PVG', 'NKG'),
            ('PVG', 'WUH'), ('PVG', 'XMN'), ('PVG', 'CGO'), ('PVG', 'HRB'),
            ('PVG', 'URC'), ('PVG', 'DLC'), ('PVG', 'SYX'), ('PVG', 'TAO'),
            ('PVG', 'CSX'), ('PVG', 'NNG'), ('PVG', 'FOC'), ('PVG', 'HFE'),
            # 广州/深圳出发
            ('CAN', 'KMG'), ('CAN', 'NKG'), ('CAN', 'WUH'), ('CAN', 'XMN'),
            ('CAN', 'SYX'), ('CAN', 'CSX'), ('CAN', 'NNG'), ('CAN', 'FOC'),
            ('SZX', 'PEK'), ('SZX', 'PVG'), ('SZX', 'CAN'), ('SZX', 'CTU'),
            ('SZX', 'CKG'), ('SZX', 'HGH'), ('SZX', 'XMN'), ('SZX', 'SYX'),
            # 成都/重庆出发
            ('CTU', 'HGH'), ('CTU', 'XIY'), ('CTU', 'KMG'), ('CTU', 'NKG'),
            ('CTU', 'WUH'), ('CTU', 'XMN'), ('CTU', 'URC'), ('CTU', 'SYX'),
            ('CKG', 'HGH'), ('CKG', 'XIY'), ('CKG', 'KMG'), ('CKG', 'NKG'),
            ('CKG', 'WUH'), ('CKG', 'XMN'), ('CKG', 'SYX'), ('CKG', 'CSX'),
            # 其他重要连接
            ('HGH', 'CTU'), ('HGH', 'CKG'), ('HGH', 'XIY'), ('HGH', 'KMG'),
            ('XIY', 'KMG'), ('XIY', 'URC'), ('KMG', 'SYX'), ('NKG', 'CTU'),
        ]
        
        for orig, dest in china_domestic_pairs[:80]:
            if orig in self.airports and dest in self.airports:
                routes.append(self._create_route(orig, dest, '中国'))
        
        # 2. 亚洲国际航线（60条）
        asia_international_pairs = [
            # 中国-日本
            ('PEK', 'NRT'), ('PEK', 'HND'), ('PEK', 'KIX'), ('PVG', 'NRT'),
            ('PVG', 'HND'), ('PVG', 'KIX'), ('PVG', 'NGO'), ('CAN', 'NRT'),
            ('CAN', 'KIX'), ('SZX', 'NRT'), ('HGH', 'NRT'), ('CTU', 'NRT'),
            # 中国-韩国
            ('PEK', 'ICN'), ('PEK', 'PUS'), ('PVG', 'ICN'), ('PVG', 'GMP'),
            ('CAN', 'ICN'), ('SZX', 'ICN'), ('HGH', 'ICN'), ('CTU', 'ICN'),
            # 中国-东南亚
            ('PEK', 'SIN'), ('PEK', 'BKK'), ('PEK', 'KUL'), ('PVG', 'SIN'),
            ('PVG', 'BKK'), ('PVG', 'KUL'), ('CAN', 'SIN'), ('CAN', 'BKK'),
            ('CAN', 'KUL'), ('CAN', 'MNL'), ('CAN', 'CGK'), ('SZX', 'SIN'),
            ('SZX', 'BKK'), ('CTU', 'BKK'), ('CKG', 'BKK'),
            # 中国-香港/台北
            ('PEK', 'HKG'), ('PVG', 'HKG'), ('CAN', 'HKG'), ('CTU', 'HKG'),
            ('PEK', 'TPE'), ('PVG', 'TPE'), ('CAN', 'TPE'), ('SZX', 'TPE'),
            # 中国-南亚/中东
            ('PEK', 'DEL'), ('PVG', 'DEL'), ('PEK', 'DXB'), ('PVG', 'DXB'),
            ('CAN', 'DXB'), ('PEK', 'DOH'), ('PVG', 'DOH'),
            # 亚洲内部其他
            ('NRT', 'SIN'), ('NRT', 'BKK'), ('NRT', 'ICN'), ('ICN', 'SIN'),
            ('ICN', 'BKK'), ('HKG', 'SIN'), ('HKG', 'BKK'), ('HKG', 'TPE'),
            ('SIN', 'BKK'), ('SIN', 'KUL'), ('SIN', 'CGK'),
        ]
        
        for orig, dest in asia_international_pairs[:60]:
            if orig in self.airports and dest in self.airports:
                routes.append(self._create_route(orig, dest, '亚洲'))
        
        # 3. 跨太平洋航线（40条）
        transpacific_pairs = [
            # 中国-北美
            ('PEK', 'LAX'), ('PEK', 'SFO'), ('PEK', 'JFK'), ('PEK', 'ORD'),
            ('PEK', 'SEA'), ('PEK', 'YVR'), ('PVG', 'LAX'), ('PVG', 'SFO'),
            ('PVG', 'JFK'), ('PVG', 'SEA'), ('CAN', 'LAX'), ('CAN', 'SFO'),
            ('CAN', 'YVR'), ('CTU', 'LAX'), ('SZX', 'LAX'),
            # 日本-北美
            ('NRT', 'LAX'), ('NRT', 'SFO'), ('NRT', 'JFK'), ('NRT', 'ORD'),
            ('NRT', 'SEA'), ('NRT', 'YVR'), ('HND', 'LAX'), ('HND', 'SFO'),
            # 韩国-北美
            ('ICN', 'LAX'), ('ICN', 'SFO'), ('ICN', 'JFK'), ('ICN', 'SEA'),
            # 东南亚-北美
            ('SIN', 'LAX'), ('SIN', 'SFO'), ('SIN', 'JFK'), ('BKK', 'LAX'),
            # 大洋洲-北美
            ('SYD', 'LAX'), ('SYD', 'SFO'), ('SYD', 'YVR'), ('MEL', 'LAX'),
            ('AKL', 'LAX'), ('AKL', 'SFO'), ('BNE', 'LAX'),
            # 香港/台北-北美
            ('HKG', 'LAX'), ('HKG', 'SFO'), ('TPE', 'LAX'),
        ]
        
        for orig, dest in transpacific_pairs[:40]:
            if orig in self.airports and dest in self.airports:
                routes.append(self._create_route(orig, dest, '跨太平洋'))
        
        # 4. 跨大西洋航线（40条）
        transatlantic_pairs = [
            # 美国-欧洲
            ('JFK', 'LHR'), ('JFK', 'CDG'), ('JFK', 'FRA'), ('JFK', 'AMS'),
            ('JFK', 'MAD'), ('JFK', 'FCO'), ('JFK', 'MXP'), ('JFK', 'ZRH'),
            ('JFK', 'VIE'), ('JFK', 'ATH'), ('JFK', 'IST'), ('JFK', 'SVO'),
            ('EWR', 'LHR'), ('EWR', 'CDG'), ('EWR', 'FRA'), ('EWR', 'AMS'),
            ('LAX', 'LHR'), ('LAX', 'CDG'), ('LAX', 'FRA'), ('LAX', 'AMS'),
            ('ORD', 'LHR'), ('ORD', 'CDG'), ('ORD', 'FRA'), ('ORD', 'AMS'),
            ('SFO', 'LHR'), ('SFO', 'CDG'), ('SFO', 'FRA'), ('SFO', 'AMS'),
            ('ATL', 'LHR'), ('ATL', 'CDG'), ('ATL', 'AMS'), ('BOS', 'LHR'),
            ('MIA', 'LHR'), ('MIA', 'MAD'), ('IAH', 'LHR'), ('DEN', 'LHR'),
            # 加拿大-欧洲
            ('YYZ', 'LHR'), ('YYZ', 'CDG'), ('YVR', 'LHR'), ('YUL', 'CDG'),
        ]
        
        for orig, dest in transatlantic_pairs[:40]:
            if orig in self.airports and dest in self.airports:
                routes.append(self._create_route(orig, dest, '跨大西洋'))
        
        # 5. 亚欧航线（30条）
        asia_europe_pairs = [
            # 中国-欧洲
            ('PEK', 'LHR'), ('PEK', 'CDG'), ('PEK', 'FRA'), ('PEK', 'AMS'),
            ('PEK', 'MAD'), ('PEK', 'FCO'), ('PEK', 'SVO'), ('PEK', 'IST'),
            ('PVG', 'LHR'), ('PVG', 'CDG'), ('PVG', 'FRA'), ('PVG', 'AMS'),
            ('PVG', 'FCO'), ('CAN', 'LHR'), ('CAN', 'CDG'), ('CAN', 'AMS'),
            # 中东-欧洲
            ('DXB', 'LHR'), ('DXB', 'CDG'), ('DXB', 'FRA'), ('DXB', 'AMS'),
            ('DOH', 'LHR'), ('DOH', 'CDG'), ('DOH', 'FRA'),
            # 亚洲其他-欧洲
            ('SIN', 'LHR'), ('SIN', 'CDG'), ('SIN', 'FRA'), ('HKG', 'LHR'),
            ('DEL', 'LHR'), ('BOM', 'LHR'), ('NRT', 'LHR'),
        ]
        
        for orig, dest in asia_europe_pairs[:30]:
            if orig in self.airports and dest in self.airports:
                routes.append(self._create_route(orig, dest, '亚欧'))
        
        # 6. 美国国内航线（20条）
        us_domestic_pairs = [
            ('JFK', 'LAX'), ('JFK', 'SFO'), ('JFK', 'ORD'), ('JFK', 'MIA'),
            ('JFK', 'ATL'), ('LAX', 'SFO'), ('LAX', 'LAS'), ('LAX', 'SEA'),
            ('LAX', 'ORD'), ('SFO', 'SEA'), ('SFO', 'LAS'), ('ORD', 'ATL'),
            ('ORD', 'DEN'), ('ATL', 'MIA'), ('DFW', 'LAX'), ('DFW', 'ORD'),
            ('PHX', 'LAX'), ('DEN', 'LAX'), ('SEA', 'SAN'), ('BOS', 'SFO'),
        ]
        
        for orig, dest in us_domestic_pairs:
            if orig in self.airports and dest in self.airports:
                routes.append(self._create_route(orig, dest, '美国'))
        
        # 7. 欧洲内部航线（15条）
        europe_domestic_pairs = [
            ('LHR', 'CDG'), ('LHR', 'FRA'), ('LHR', 'AMS'), ('LHR', 'MAD'),
            ('LHR', 'FCO'), ('CDG', 'FRA'), ('CDG', 'AMS'), ('CDG', 'BCN'),
            ('FRA', 'MXP'), ('AMS', 'BCN'), ('MAD', 'FCO'), ('IST', 'LHR'),
            ('SVO', 'LHR'), ('ATH', 'CDG'), ('VIE', 'LHR'),
        ]
        
        for orig, dest in europe_domestic_pairs:
            if orig in self.airports and dest in self.airports:
                routes.append(self._create_route(orig, dest, '欧洲'))
        
        # 8. 其他重要航线（15条）
        other_pairs = [
            # 南美
            ('GRU', 'JFK'), ('GRU', 'LHR'), ('GRU', 'CDG'), ('EZE', 'MAD'),
            ('SCL', 'LAX'), ('LIM', 'LAX'), ('BOG', 'MIA'),
            # 非洲
            ('JNB', 'LHR'), ('CAI', 'CDG'), ('ADD', 'DXB'),
            # 大洋洲内部
            ('SYD', 'MEL'), ('SYD', 'BNE'), ('SYD', 'PER'), ('SYD', 'AKL'),
            ('MEL', 'AKL'),
        ]
        
        for orig, dest in other_pairs:
            if orig in self.airports and dest in self.airports:
                routes.append(self._create_route(orig, dest, '其他'))
        
        return routes[:300]  # 确保正好300条
    
    def _create_route(self, origin_code: str, dest_code: str, category: str) -> dict:
        """创建单条航线记录"""
        orig_data = self.airports[origin_code]
        dest_data = self.airports[dest_code]
        
        orig_lat, orig_lon = orig_data[0], orig_data[1]
        dest_lat, dest_lon = dest_data[0], dest_data[1]
        orig_city = orig_data[2]
        dest_city = dest_data[2]
        
        # 计算距离
        distance = self.calculate_distance(orig_lat, orig_lon, dest_lat, dest_lon)
        
        # 确定航路点数量
        num_waypoints = self.determine_waypoint_count(distance)
        
        # 生成航路点
        waypoints = self.calculate_waypoints_realistic(
            orig_lat, orig_lon, dest_lat, dest_lon, num_waypoints
        )
        
        # 创建路由记录
        route = {
            'origin_code': origin_code,
            'origin_lat': orig_lat,
            'origin_lon': orig_lon,
            'dest_code': dest_code,
            'dest_lat': dest_lat,
            'dest_lon': dest_lon,
            'route_name': f"{orig_city}-{dest_city}",
            'category': category,
            'distance_km': round(distance, 1),
        }
        
        # 添加航路点（最多10个）
        for i in range(10):
            if i < len(waypoints):
                route[f'waypoint{i+1}_lat'] = waypoints[i][0]
                route[f'waypoint{i+1}_lon'] = waypoints[i][1]
            else:
                route[f'waypoint{i+1}_lat'] = None
                route[f'waypoint{i+1}_lon'] = None
        
        return route


def main():
    """主函数：生成航线数据并保存"""
    print("="*80)
    print("真实航线数据生成器")
    print("="*80)
    print("\n正在初始化航线生成器...")
    
    generator = RouteGenerator()
    
    print("正在生成300条全球代表性航线...")
    routes = generator.generate_routes()
    
    print(f"✓ 成功生成 {len(routes)} 条航线")
    
    # 统计信息
    print("\n航线分类统计:")
    categories = {}
    for route in routes:
        cat = route['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"  {cat:12s}: {count:3d} 条")
    
    # 距离统计
    distances = [r['distance_km'] for r in routes]
    print(f"\n距离统计:")
    print(f"  最短: {min(distances):.0f} 公里")
    print(f"  最长: {max(distances):.0f} 公里")
    print(f"  平均: {np.mean(distances):.0f} 公里")
    print(f"  总计: {sum(distances):.0f} 公里")
    
    # 航路点统计
    waypoint_counts = []
    for route in routes:
        count = sum(1 for i in range(1, 11) if route.get(f'waypoint{i}_lat') is not None)
        waypoint_counts.append(count)
    
    print(f"\n航路点统计:")
    print(f"  最少: {min(waypoint_counts)} 个")
    print(f"  最多: {max(waypoint_counts)} 个")
    print(f"  平均: {np.mean(waypoint_counts):.1f} 个")
    
    # 保存到CSV
    output_file = 'waypoints.csv'
    print(f"\n正在保存到 {output_file}...")
    
    # 定义列顺序
    columns = ['origin_code', 'origin_lat', 'origin_lon', 
               'dest_code', 'dest_lat', 'dest_lon']
    for i in range(1, 11):
        columns.extend([f'waypoint{i}_lat', f'waypoint{i}_lon'])
    columns.extend(['route_name', 'category', 'distance_km'])
    
    df = pd.DataFrame(routes, columns=columns)
    df.to_csv(output_file, index=False, float_format='%.4f')
    
    print(f"✓ 数据已保存到 {output_file}")
    print(f"  文件大小: {len(open(output_file).read())/1024:.1f} KB")
    print(f"  数据行数: {len(df)+1} 行（含标题）")
    
    print("\n" + "="*80)
    print("数据生成完成！")
    print("="*80)
    
    return df


if __name__ == '__main__':
    main()

