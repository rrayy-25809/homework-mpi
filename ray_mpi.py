from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
#TODO:광원 움직이게 달 빠르게 움직이게
# 벡터 정규화 함수
def normalize(vector):
    return vector / np.linalg.norm(vector)

# 벡터를 주어진 축에 대해 반사시키는 함수
def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

# 구와의 교차점을 계산하는 함수
def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b **2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

# 가장 가까운 교차점을 찾는 함수
def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

# 레이 트레이싱 메인 함수
def ray_tracing(x, y):
    # 화면은 원점에 있음
    pixel = np.array([x, y, 0])
    origin = camera
    direction = normalize(pixel - origin)
    color = np.zeros((3))
    reflection = 1
    for k in range(max_depth):
        # 교차점 확인
        nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
        if nearest_object is None:
            break
        intersection = origin + min_distance * direction
        normal_to_surface = normalize(intersection - nearest_object['center'])
        shifted_point = intersection + 1e-5 * normal_to_surface
        intersection_to_light = normalize(light['position'] - shifted_point)
        _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
        intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
        is_shadowed = min_distance < intersection_to_light_distance
        if is_shadowed:
            break
        illumination = np.zeros((3))
        # 주변광
        illumination += nearest_object['ambient'] * light['ambient']
        # 확산광
        illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface)
        # 반사광
        intersection_to_camera = normalize(camera - intersection)
        H = normalize(intersection_to_light + intersection_to_camera)
        illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4)
        # 반사
        color += reflection * illumination
        reflection *= nearest_object['reflection']
        origin = shifted_point
        direction = reflected(direction, normal_to_surface)
    return color

# 프로그램 실행 시간 측정 시작
all_start_time = MPI.Wtime()

# 최대 반사 깊이
max_depth = 3

# 이미지 크기와 카메라 및 광원 설정
width = 1200
height = 800
camera = np.array([0, 0, 0.5])

for count in range(60):
    start_time = MPI.Wtime()

    light = { 'position': np.array([5*math.sin(2*math.pi * count/60), 5, 5*math.cos(2*math.pi * count/60)]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

    # 씬에 있는 객체들 정의
    objects = [
        { 'center': np.array([0, 0.05*math.sin(2*math.pi * count/60), 0]), 'radius': 0.2, 'ambient': np.array([0, 0, 0.7]), 'diffuse': np.array([0.7, 1, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 120, 'reflection': 0.5 },
        { 'center': np.array([0.4*math.cos(2*math.pi * 5*count/60), 0, 0.4*math.sin(2*math.pi * 5*count/60)]), 'radius': 0.03, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.2 },
        { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
    ]

    #기본 mpi변수들 선언
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # 화면 비율과 스크린 크기 설정
    ratio = float(width) / height 
    screen = (-1, 1 / ratio, 1, -1 / ratio)
    image = np.empty([height, width, 3]) if rank == 0 else None

    N = height // size + int(height % size > rank)
    start = comm.scan(N)-N

    # 각 프로세스가 처리할 행 수를 계산
    # height // size는 각 프로세스가 처리할 기본 행 수를 의미하고,
    # height % size는 나머지 행을 각 프로세스에 골고루 분배하기 위한 조건입니다.
    rows_per_process = [height // size + (1 if i < height % size else 0) for i in range(size)]

    # 각 프로세스가 보낼 픽셀 수를 계산
    # 각 픽셀은 RGB로 구성되어 있어 각 픽셀마다 3개의 값이 존재하므로 width * 3을 곱합니다.
    sendcounts = [rows * width * 3 for rows in rows_per_process]

    # displs 배열 계산 (각 프로세스의 데이터 시작 위치)
    displs = [0] * size
    for i in range(1, size):
        displs[i] = displs[i - 1] + sendcounts[i - 1]

    Y = np.linspace(screen[1], screen[3], height)[start:start+N]
    X = np.linspace(screen[0], screen[2], width)

    rank_image = np.zeros((N, width, 3))

    for i, y in enumerate(Y):
        for j, x in enumerate(X):
            color = ray_tracing(x, y) 
            rank_image[i, j] = np.clip(color, 0, 1)

    comm.Gatherv(sendbuf=rank_image,recvbuf=(image, sendcounts, displs, MPI.DOUBLE),root=0)
    if rank==0:
        # 결과 이미지를 파일로 저장
        plt.imsave(f'images/{count}.png', image)
        end_time = MPI.Wtime()
        print(f"process {count + 1} is end at {end_time-start_time}")

if rank==0:
    images = []
    for image in range(60):
        images.append(Image.open(f'./images/{image}.png'))

    # 프로그램 실행 시간 측정 종료 (원본 파일의 시간 약 26초)
    all_end_time = MPI.Wtime()
    images[0].save('orbit.gif', save_all=True, append_images=images[1:], duration=100, loop =0)
    print(f"gif is generated at {all_end_time - all_start_time}")