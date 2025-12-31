"""
导航数学工具库
提供大圆航线计算、距离、航向等导航功能
"""
import math


class NavUtils:
    """导航工具类"""
    R_EARTH = 6371000.0  # 地球半径（米）

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        计算两点间的大圆距离（Haversine公式）
        
        Args:
            lat1, lon1: 起点经纬度（度）
            lat2, lon2: 终点经纬度（度）
            
        Returns:
            距离（米）
        """
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * NavUtils.R_EARTH * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        计算从起点到终点的航向角
        
        Args:
            lat1, lon1: 起点经纬度（度）
            lat2, lon2: 终点经纬度（度）
            
        Returns:
            航向角（度，0-360）
        """
        y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
        x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
            math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1))
        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360) % 360
    
    @staticmethod
    def intermediate_point(lat1: float, lon1: float, lat2: float, lon2: float, fraction: float) -> tuple:
        """
        计算大圆航线上的中间点
        
        Args:
            lat1, lon1: 起点经纬度（度）
            lat2, lon2: 终点经纬度（度）
            fraction: 比例（0-1，0为起点，1为终点）
            
        Returns:
            (纬度, 经度) 元组
        """
        phi1, lambda1 = math.radians(lat1), math.radians(lon1)
        phi2, lambda2 = math.radians(lat2), math.radians(lon2)
        
        # 角度距离
        d = 2 * math.asin(math.sqrt(math.sin((phi2-phi1)/2)**2 + 
             math.cos(phi1)*math.cos(phi2)*math.sin((lambda2-lambda1)/2)**2))
        
        if d == 0:
            return lat1, lon1
        
        a = math.sin((1-fraction)*d) / math.sin(d)
        b = math.sin(fraction*d) / math.sin(d)
        
        x = a * math.cos(phi1) * math.cos(lambda1) + b * math.cos(phi2) * math.cos(lambda2)
        y = a * math.cos(phi1) * math.sin(lambda1) + b * math.cos(phi2) * math.sin(lambda2)
        z = a * math.sin(phi1) + b * math.sin(phi2)
        
        phi_i = math.atan2(z, math.sqrt(x**2 + y**2))
        lambda_i = math.atan2(y, x)
        
        return math.degrees(phi_i), math.degrees(lambda_i)

