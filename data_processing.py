#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Author: shi yao <euler_shiyao@foxmail.com>
@Create Time: 2023/1/25 22:58
python version: python 3.8.12
"""


import time
from typing import Tuple, List, Any
import matplotlib.pyplot as plt
import os
import micasense.metadata as metadata
import micasense.plotutils as plotutils
import micasense.utils as msutils
from numpy import ndarray
from CFG import CFG
import datetime as dt
import utils
import micasense.capture as capture
import glob
import micasense.imageutils as imageutils
import cv2
import numpy as np
import logging
import micasense.imageset as imageset
import pandas as pd
import json
import numba
import warnings
from numba import njit, typeof, typed, types
warnings.filterwarnings("ignore")
from numba import jit
from numba.typed import Dict
from tqdm import tqdm
from sys import getsizeof as gso
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing.pool import ThreadPool
from concurrent.futures import ProcessPoolExecutor
import cvxopt
import Metashape
import shutil
from osgeo import gdal, gdalconst
Metashape.License().activateOffline("T7BX9-S9XJC-SBBTV-3N6RO-2TH33")

class DataProcessing(CFG):
    def __init__(self, args: list or str, screen_show_info: bool = True):
        super().__init__(args, screen_show_info)
        self.exif_tool_Path = os.environ.get('exif_tool_path')
        self.SET_ = {  # 不同SET、不同位置的reference图片的编号
            'SET1': [113, 114, 115],
            'SET2': [380, 381, 382],
            'SET3': [236, 237, 238],
            'SET4': [450, 451, 452],
        }
        self.images_dir = self.path['images_dir']  # raw的路径
        self.BRDF_images_dir = self.path['BRDF_images_dir']  # BRDF的路径
        self.GELM_images_dir = self.path['GELM_images_dir']  # GELM的路径
        self.images_dir = self.path['images_dir']  # raw的路径
        # reference图片的路径、json路径、ID
        self.ref_img_path_dict, self.ref_json_path_dict, self.ref_id = self.get_reference_img_path()
        # overlap图片的路径、ID
        self.overlap_img_path_dict, self.overlap_id = self.get_overlap_img_path()
        self.ref_type = self.to_list(self.param['ref_name'])  # 板子的颜色
        self.ref_type_json_num = self.gen_ref_type_json_num()
        self.SET_REF = self.gen_SET_REF_dict()
        self.reference_length = len(self.ref_img_path_dict)  # reference图片的个数
        self.all_num = len(os.listdir(self.images_dir))  # 全部图片的个数
        self.ID_max = int(os.listdir(self.images_dir)[-1].split('_')[1])  + 1 # 最大的ID数
        self.overlapping_length = self.all_num - self.reference_length  # overlapping图片的个数


    def gen_dls_df(self):
        ID = []
        for pp in os.listdir(self.images_dir):
            logging.debug(fr"正在获取{pp}的图像ID")
            ID.append(int(pp.split('_')[1]))
        df_DLS = self.micasense_DLS(self.images_dir)
        df_DLS['ID'] = np.array(ID)
        df_DLS.to_csv(self.save_path['DLS'])
        return df_DLS

    def reference_processing(self):
        L_reference, R_reference, ID_reference = self.get_reference_points()
        reference_points_num = len(L_reference)  # reference点的个数
        logging.debug(fr"reference点共有{reference_points_num}个")
        En = np.zeros((reference_points_num, self.ID_max))  # 初始化reference方程
        k_temp = 0
        for n in ID_reference:
            En[k_temp, int(n)] = 1
            k_temp += 1
        result_n = L_reference - R_reference + np.log(np.pi)
        np.save(self.save_path['En'], En)
        np.save(self.save_path['result_n'], result_n)
        return En, result_n

    def sv_tif(self, src_file: str, des_file: str, new_data: ndarray):
        # shutil.copy(src_file, des_file)
        # ds = gdal.Open(des_file, gdal.GA_Update)
        #
        # ds.GetRasterBand(1).WriteArray(new_data)
        # del ds
        dataset = gdal.Open(src_file, gdalconst.GA_ReadOnly)
        img_width = dataset.RasterXSize
        img_height = dataset.RasterYSize
        adf_GeoTransform = dataset.GetGeoTransform()
        im_Proj = dataset.GetProjection()
        driver = gdal.GetDriverByName("GTiff")
        datasetnew = driver.Create(des_file, img_width, img_height, 1, gdal.GDT_Float32)
        datasetnew.SetGeoTransform(adf_GeoTransform)
        datasetnew.SetProjection(im_Proj)
        band = datasetnew.GetRasterBand(1)
        band.WriteArray(new_data)
        datasetnew.FlushCache()


    def run(self):
        # df_DLS = self.gen_dls_df()
        df_DLS = pd.read_csv(self.save_path['DLS'])

        # En, result_n = self.reference_processing()
        En = np.load(self.save_path['En'])
        result_n = np.load(self.save_path['result_n'])

        # L_list, L_overlap, temp_id = self.gen_tie_points()
        # np.save(r'C:\Users\aiwei\Desktop\codes\L_overlap.npy', L_overlap)
        # np.save(r'C:\Users\aiwei\Desktop\codes\temp_id.npy', temp_id)
        # np.save(r'C:\Users\aiwei\Desktop\codes\L_list.npy', L_list)
        L_list = np.load(r'C:\Users\aiwei\Desktop\codes\L_list.npy')
        # logging.debug(L_list.shape)
        # L_overlap = np.load(r'C:\Users\aiwei\Desktop\codes\L_overlap.npy')
        # temp_id = np.load(r'C:\Users\aiwei\Desktop\codes\temp_id.npy')
        # temp_id_num = int(len(temp_id[temp_id != -1])) # tie point的个数
        # logging.debug(len(L_overlap[~np.isnan(L_overlap)]))
        # logging.debug(temp_id_num)
        # Em = np.zeros((temp_id_num, self.ID_max))  # 初始化overlap矩阵
        # logging.debug(Em.shape)
        # k_temp = 0
        # logging.debug(self.all_num)
        # result_m = np.zeros(temp_id_num)
        # for i in range(self.ID_max):
        #     for j in range(i + 1, self.ID_max, 1):
        #         if ~np.isnan(L_overlap[i, j]):
        #             Em[k_temp, i] = 1
        #             Em[k_temp, j] = -1
        #             Em[k_temp + 1, i] = 1
        #             Em[k_temp + 1, j] = -1
        #             result_m[k_temp] = L_overlap[i, j]
        #             result_m[k_temp + 1] = L_overlap[j, i]
        #             k_temp += 2
        # np.save(r'C:\Users\aiwei\Desktop\codes\Em.npy', Em)
        # np.save(r'C:\Users\aiwei\Desktop\codes\result_m.npy', result_m)
        Em = np.load(r'C:\Users\aiwei\Desktop\codes\Em.npy')
        result_m = np.load(r'C:\Users\aiwei\Desktop\codes\result_m.npy')
        result_m[np.isinf(result_m)] = np.log(0.1)
        result_m[np.isnan(result_m)] = np.log(0.1)
        L_list[np.isinf(L_list)] = np.log(0.1)



        x = self.gen_res(En, Em, result_n, result_m, df_DLS, L_list)
        logging.debug("计算完成")
        x = np.array(x[df_DLS['ID']])
        df_DLS['Estimate'] = x
        df_DLS.to_csv(r'C:\Users\aiwei\Desktop\codes\result.csv')
        df_DLS = pd.read_csv(r'C:\Users\aiwei\Desktop\codes\result.csv')

        self.draw(df_DLS['ID'], df_DLS['irr-560'].values, df_DLS['Estimate'].values, 'irr-560', 'Estimate',
                  'no bound', r'C:\Users\aiwei\Desktop\codes\no bound.png')
        self.retrieve()


    def retrieve(self):
        df_DLS = pd.read_csv(r'C:\Users\aiwei\Desktop\codes\result.csv')
        Estimate = df_DLS['Estimate'].values
        ID = df_DLS['ID'].values
        for i in tqdm(range(len(ID))):
            img_path = os.path.join(self.images_dir, f"IMG_{str(ID[i]).zfill(4)}_2.tif")
            sv_path = os.path.join(self.GELM_images_dir, f"IMG_{str(ID[i]).zfill(4)}_2.tif")
            # sv_path = os.path.join(self.BRDF_images_dir, f"IMG_{str(ID[i]).zfill(4)}_2.tif")

            L = self.get_radiance_from_raw(img_path)
            # R = np.pi * L / df_DLS['irr-475'].values[i]
            R = np.pi * L / Estimate[i]
            # k = 3 if i > 3 else 1
            # R = np.round(R * 65535 * k)
            # R[R > 65535] = 65535
            # R[R < 0] = 0
            # R = R.astype(np.uint16)
            self.sv_tif(img_path, sv_path, R)


    @staticmethod
    def draw(x: ndarray, y1: ndarray, y2: ndarray, y1_name: str, y2_name: str, title: str, sv_path: str):
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)
        plt.plot(x, y1, markerfacecolor='r', label=y1_name)
        plt.plot(x, y2, markerfacecolor='b', label=y2_name)
        ax.set_xticks(x[::20])
        plt.title(title)
        plt.legend()
        plt.savefig(sv_path)
        plt.show()

    def gen_res(self, En: ndarray, Em: ndarray, result_n: ndarray, result_m: ndarray, pp: pd.DataFrame,
                L_list: ndarray) -> ndarray:
        # 文献1
        # A1, A2, B1, B2 = Em, En, result_m, result_n
        # Z1 = np.zeros((len(result_n), self.ID_max))
        # Z2 = np.zeros((len(result_m), self.ID_max))
        # Z1_ = np.zeros((len(result_n)))
        # Z2_ = np.zeros((len(result_m)))
        # P1 = np.r_[A1, Z1]
        # P2 = np.r_[A2, Z2]
        # Q1 = np.r_[B1, Z1_]
        # Q2 = np.r_[B2, Z2_]
        # alpha = 100
        # k1 = (np.dot(P1.T, P1) + alpha * np.dot(P2.T, P2))
        # k2 = np.dot(P1.T, Q1) + alpha * np.dot(P2.T, Q2)
        # x = np.dot(np.linalg.pinv(k1), k2)
        # x = np.exp(x)
        # return x

        # # 文献1

        if self.brdf:
            A1, A2, B1, B2 = Em, En, result_m, result_n
            logging.debug(len(result_m[np.isnan(result_m)]))
            Z1 = np.zeros((len(result_n), self.ID_max))
            Z2 = np.zeros((len(result_m), self.ID_max))
            Z1_ = np.zeros((len(result_n)))
            Z2_ = np.zeros((len(result_m)))
            P1 = np.r_[A1, Z1]
            P2 = np.r_[A2, Z2]
            Q1 = np.r_[B1, Z1_]
            Q2 = np.r_[B2, Z2_]
            alpha = 0.01
            k1 = (np.dot(P1.T, P1) + alpha * np.dot(P2.T, P2))
            k2 = np.dot(P1.T, Q1) + alpha * np.dot(P2.T, Q2)
            P = cvxopt.matrix((np.dot(k1.T, k1) + np.dot(k1.T, k1).T) / 2)
            q = cvxopt.matrix(-1 * np.dot(k1.T, k2))
            G = cvxopt.matrix(np.r_[np.eye(self.ID_max) * -1, np.eye(self.ID_max) * 1,
                                    np.eye(self.ID_max) * 1, np.eye(self.ID_max) * -1])
            hh1 = np.ones(self.ID_max)
            hh2 = np.ones(self.ID_max)
            hh3 = np.ones(self.ID_max)
            hh4 = np.ones(self.ID_max)
            # hh1[pp['ID']] = -1 * np.log(pp['irr-560'].values * 0.5)
            # hh1[pp['ID']] = -1 * np.log(0.4)
            hh1[pp['ID']] = -1 * L_list[pp['ID']]
            # hh2[pp['ID']] = 1 * np.log(pp['irr-560'].values * 1.5)
            hh2[pp['ID']] = L_list[pp['ID']] - np.log(0.001)
            # hh2[pp['ID']] = 1 * np.log(1.4)
            hh3[pp['ID']] = np.log(2)
            hh4[pp['ID']] = -1 * np.log(0.4)
            cvxopt.solvers.options['show_progress'] = False
            # h = cvxopt.matrix(hh1)
            h = cvxopt.matrix(np.r_[hh1, hh2, hh3, hh4])
            # h = cvxopt.matrix(np.ones(self.ID_max) * np.log(np.nanmax(pp)))
            sol = cvxopt.solvers.qp(P, q, G, h)
            x = sol['x']
            x = np.exp(x)
            x = utils.filter_ndarray(utils.filter_ndarray(x, True), True)
            return x

        # EL、MIEL
        # result = np.r_[result_n, result_m]
        # E = np.r_[En, Em]
        # x = np.dot(np.linalg.pinv(E), result)
        # x = np.exp(x)
        # x[x > 1.5] = x[x > 1.5] / 1000
        #
        # x[x < 0.25] = x[x < 0.25] * 10
        # # x[x > 1.5] = x[x > 1.5] / 10
        # return x

        # MIcEL
        result = np.r_[result_n, result_m]
        E = np.r_[En, Em]
        P = cvxopt.matrix(np.dot(E.T, E))
        q = cvxopt.matrix(-1 * np.dot(E.T, result))
        G = cvxopt.matrix(np.r_[np.eye(self.ID_max) * -1, np.eye(self.ID_max) * 1])
        hh1 = np.ones(self.ID_max)
        hh1[pp['ID']] = -1 * np.log(pp['irr-560'].values * 0.3)
        # hh1[pp['ID']] = -1 * np.log(0.4)
        # hh1[hh1 > np.log(1.5)] = np.log(1.5)
        hh2 = np.ones(self.ID_max)
        # hh2[pp['ID']] = np.log(1)
        # hh2[pp['ID']] = np.log(pp['irr-560'].values * 1.4)
        hh2[pp['ID']] = np.log(1.4)
        # hh2[pp['ID']] = np.log(2)
        h = cvxopt.matrix(np.r_[hh1, hh2])
        # h = cvxopt.matrix(np.ones(self.ID_max) * np.log(np.nanmax(pp)))
        # cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P,q,G,h)
        x = sol['x']
        x = np.exp(x)
        return x

    @staticmethod
    def get_tie_points(camera1: Metashape.Camera, camera2: Metashape.Camera, chunk: Metashape.Chunk, name: str=''):
        if (not camera1.transform) or (not camera2.transform):
            return None

        projections = chunk.tie_points.projections
        points = chunk.tie_points.points
        tracks = chunk.tie_points.tracks
        npoints = len(points)
        point_ids = [-1] * len(chunk.tie_points.tracks)
        for point_id in range(0, npoints):
            point_ids[points[point_id].track_id] = point_id
        camera_matches_valid = dict()

        tie_points = dict()
        tie_points[camera1] = dict()
        tie_points[camera2] = dict()

        for camera in [camera1, camera2]:
            T = camera.transform.inv()
            calib = camera.sensor.calibration
            valid_matches = set()
            for proj in projections[camera]:
                track_id = proj.track_id
                point_id = point_ids[track_id]

                if point_id < 0:
                    continue
                if not points[point_id].valid:  # skipping invalid points
                    continue
                valid_matches.add(point_id)
                tie_points[camera][point_id] = proj.coord

            camera_matches_valid[camera] = valid_matches
        valid = camera_matches_valid[camera1].intersection(camera_matches_valid[camera2])

        path = os.path.join(r'C:\Users\aiwei\Desktop\1', name)
        if name != '':
            file = open(path, "wt")
            file.write("#point_id,camera1,camera2,cam1_x,cam1_y,cam2_x,cam2_y\n")
        tie_point_output = []
        for point_id in valid:
            tie_point_output.append((tie_points[camera1][point_id], tie_points[camera2][point_id]))
            if not len(tie_point_output):
                return None
            if name != '':
                file.write(str(point_id) + "," + camera1.label + "," + camera2.label
                           + ",{:.2f},{:.2f},{:.2f},{:.2f}\n".format(tie_points[camera1][point_id].x,
                                                                     tie_points[camera1][point_id].y,
                                                                     tie_points[camera2][point_id].x,
                                                                     tie_points[camera2][point_id].y))
        if name != '':
            file.close()
        return tie_point_output


    def get_chunk(self, re_run: bool=False) -> Metashape.Document.chunk:
        doc = Metashape.Document()
        if os.path.exists(self.save_path['chunk']) and not re_run:
            doc.open(self.save_path['chunk'])
            return doc.chunk
        path_list =[]
        for img_name in os.listdir(self.images_dir):
            path_list.append(os.path.join(self.images_dir, img_name))
        chunk = doc.addChunk()
        chunk.addPhotos(path_list)
        chunk.matchPhotos()
        chunk.alignCameras()
        doc.save(self.save_path['chunk'])
        return doc.chunk

    def gen_tie_points_(self) -> Tuple[ndarray, ndarray, ndarray]:
        temp_id = []
        chunk = self.get_chunk()  # metashape导入全部图片
        num = len(chunk.cameras)  # 总共有多少图
        for i in range(num):  # 任取2个组合
            for j in range(i + 1, num, 1):
                temp_id.append([i, j])
        # radiance矩阵，radiance[i, j]存储img_i和img_j选择的2个tie points中radiance最大的那一个，radiance[j, i]存小的
        L_overlap = np.zeros((self.ID_max, self.ID_max)) * np.nan  # radiance
        points_list = []
        id_list = []

        for q in tqdm(range(len(temp_id))):
            i, j = temp_id[q]  # 第i张camera和第j张camera
            # img_path_i = os.path.join(self.images_dir, chunk.cameras[i].label + '.tif')
            # img_path_j = os.path.join(self.images_dir, chunk.cameras[j].label + '.tif')
            i_ = int(chunk.cameras[i].label.split('_')[1])  # 第i张图的图像编号
            j_ = int(chunk.cameras[j].label.split('_')[1])
            # common_tiepoints = self.get_tie_points(chunk.cameras[i], chunk.cameras[j], chunk)
            # np.save(fr"C:\Users\aiwei\Desktop\temp\common_tiepoints_{i}_{j}_{i_}_{j_}.npy", common_tiepoints)
            try:
                common_tiepoints = np.load(fr"C:\Users\aiwei\Desktop\temp\common_tiepoints_{i}_{j}_{i_}_{j_}.npy")
            except FileNotFoundError:
                common_tiepoints = np.load(fr"G:\temp\common_tiepoints_{i}_{j}_{i_}_{j_}.npy")
            if (common_tiepoints is None) or (len(common_tiepoints) <= 2500):
                temp_id[q] = [-1, -1]
                continue
            # if (common_tiepoints is None) or (len(common_tiepoints) == 0):
            #     temp_id[q] = [-1, -1]
            #     continue
            # ptsA, ptsB, img_i, img_j = self.gen_good_tie_points(i, j, common_tiepoints, img_path_i, img_path_j)
            # ptsA = np.save(fr'C:\Users\aiwei\Desktop\temp1\ptsA_{i}_{j}_{i_}_{j_}.npy', ptsA)
            # ptsB = np.save(fr'C:\Users\aiwei\Desktop\temp1\ptsB_{i}_{j}_{i_}_{j_}.npy', ptsB)
            # img_i = np.save(fr'C:\Users\aiwei\Desktop\temp1\img_i_{i}_{j}_{i_}_{j_}.npy', img_i)
            # img_j = np.save(fr'C:\Users\aiwei\Desktop\temp1\img_j_{i}_{j}_{i_}_{j_}.npy', img_j)
            try:
                ptsA = np.load(fr'G:\temp1\ptsA_{i}_{j}_{i_}_{j_}.npy')
                if len(ptsA) < 2500:
                    continue
                ptsB = np.load(fr'G:\temp1\ptsB_{i}_{j}_{i_}_{j_}.npy')
                img_i = np.load(fr'G:\temp1\img_i_{i}_{j}_{i_}_{j_}.npy')
                img_j = np.load(fr'G:\temp1\img_j_{i}_{j}_{i_}_{j_}.npy')
            except FileNotFoundError:
                temp_id[q] = [-1, -1]
                continue
            ptsA = np.round(ptsA).astype(int)
            ptsB = np.round(ptsB).astype(int)
            L_i = img_i[ptsA[:, 1], ptsA[:, 0]]
            L_j = img_j[ptsB[:, 1], ptsB[:, 0]]
            L_i = utils.filter_ndarray(L_i)
            L_j = utils.filter_ndarray(L_j)
            L_j[L_j < 0] = np.nan
            L_i[L_i < 0] = np.nan
            L_i[np.isnan(L_j)] = np.nan
            L_j[np.isnan(L_i)] = np.nan
            L_max_i, L_min_i = np.log(np.nanmean(L_i)), np.log(np.nanmean(L_i))
            L_max_j, L_min_j = np.log(np.nanmean(L_j)), np.log(np.nanmean(L_j))
            L_max_diff, L_min_diff = L_max_i - L_max_j, L_min_i - L_min_j
            # 图像i和图像j交集区域的一个tie point的L(radiance)记录
            L_overlap[i_, j_] = L_max_diff  # 图像i和图像j交集区域的一个tie point的ID记录
            L_overlap[j_, i_] = L_min_diff  # 图像i和图像j交集区域的一个tie point的ID记录
            id_list.append(i_)
            id_list.append(j_)
            points_list.append(L_min_i)
            points_list.append(L_min_j)
        temp_id = np.array(temp_id)
        logging.debug(f"{self.all_num}张图，共有{len(temp_id)}种组合")
        np.save(r'C:\Users\aiwei\Desktop\codes\id_list.npy', id_list)
        id_list = np.load(r'C:\Users\aiwei\Desktop\codes\id_list.npy')
        np.save(r'C:\Users\aiwei\Desktop\codes\points_list.npy', points_list)
        points_list = np.load(r'C:\Users\aiwei\Desktop\codes\points_list.npy')
        L_list = np.zeros(self.ID_max)
        L_list[id_list] = points_list
        logging.debug(L_list.shape)
        return L_list, L_overlap, temp_id

    def gen_tie_points(self) -> Tuple[ndarray, ndarray, ndarray]:
        temp_id = []
        chunk = self.get_chunk()  # metashape导入全部图片
        num = len(chunk.cameras)  # 总共有多少图
        for i in range(num):  # 任取2个组合
            for j in range(i + 1, num, 1):
                temp_id.append([i, j])
        # radiance矩阵，radiance[i, j]存储img_i和img_j选择的2个tie points中radiance最大的那一个，radiance[j, i]存小的
        L_overlap = np.zeros((self.ID_max, self.ID_max)) * np.nan  # radiance
        points_list = []
        id_list = []

        # for q in range(len(temp_id)):
        for q in tqdm(range(len(temp_id))):
            i, j = temp_id[q]  # 第i张camera和第j张camera
            # logging.debug(fr"选择图片序列中的第{i}和第{j}张：{chunk.cameras[i].label}、{chunk.cameras[j].label}")
            img_path_i = os.path.join(self.images_dir, chunk.cameras[i].label + '.tif')
            img_path_j = os.path.join(self.images_dir, chunk.cameras[j].label + '.tif')
            i_ = int(chunk.cameras[i].label.split('_')[1])  # 第i张图的图像编号
            j_ = int(chunk.cameras[j].label.split('_')[1])
            # logging.debug(fr"图片的ID是{i_}和{j_}")
            # if abs(i_ - j_) > 3:
            #     temp_id[q] = [-1, -1]
            #     continue
            # common_tiepoints = self.get_tie_points(chunk.cameras[i], chunk.cameras[j], chunk)
            # if (common_tiepoints is None) or (len(common_tiepoints) < 400):
            #     temp_id[q] = [-1, -1]
            #     continue
            # np.save(fr"G:\2\temp\common_tiepoints_{i}_{j}_{i_}_{j_}.npy", common_tiepoints)
            try:
                common_tiepoints = np.load(fr"G:\2\temp\common_tiepoints_{i}_{j}_{i_}_{j_}.npy")
            except FileNotFoundError:
                continue
            if (common_tiepoints is None) or (len(common_tiepoints) < 400):
                temp_id[q] = [-1, -1]
                continue
            img_i = utils.get_radiance_from_raw(img_path_i)
            img_j = utils.get_radiance_from_raw(img_path_j)
            ptsA, ptsB = self.gen_good_tie_points(common_tiepoints)
            # ptsA = np.save(fr'G:\2\temp1\ptsA_{i}_{j}_{i_}_{j_}.npy', ptsA)
            # ptsB = np.save(fr'G:\2\temp1\ptsB_{i}_{j}_{i_}_{j_}.npy', ptsB)
            # try:
            #     ptsA = np.load(fr'G:\2\temp1\ptsA_{i}_{j}_{i_}_{j_}.npy')
            #     ptsB = np.load(fr'G:\2\temp1\ptsB_{i}_{j}_{i_}_{j_}.npy')
            # except FileNotFoundError:
            #     temp_id[q] = [-1, -1]
            #     continue
            max_h1, min_h1, max_w1, min_w1, max_h2, min_h2, max_w2, min_w2 = \
                utils.gen_sift_points(ptsA, ptsB, img_path_i, img_path_j)


            if max_h1 != -1:
                # img_i中两个tie point的radiance
                L_max_i, L_min_i = np.log(img_i[max_h1, max_w1]), np.log(img_i[min_h1, min_w1])
                # img_j中两个tie point的radiance
                L_max_j, L_min_j = np.log(img_j[max_h2, max_w2]), np.log(img_j[min_h2, min_w2])
                try:
                    if self.brdf:
                        img_i = self.BRDF_fix(img_path_i)
                        img_j = self.BRDF_fix(img_path_j)
                except Exception:
                    pass
                L_max_diff, L_min_diff = L_max_i - L_max_j, L_min_i - L_min_j
                # 图像i和图像j交集区域的一个tie point的L(radiance)记录
                L_overlap[i_, j_] = L_max_diff  # 图像i和图像j交集区域的一个tie point的ID记录
                L_overlap[j_, i_] = L_min_diff  # 图像i和图像j交集区域的一个tie point的ID记录
                id_list.append(i_)
                id_list.append(j_)
                points_list.append(L_min_i)
                points_list.append(L_min_j)
            else:
                temp_id[q] = [-1, -1]
        temp_id = np.array(temp_id)
        logging.debug(f"{self.all_num}张图，共有{len(temp_id)}种组合")
        np.save(r'C:\Users\aiwei\Desktop\codes\id_list.npy', id_list)
        id_list = np.load(r'C:\Users\aiwei\Desktop\codes\id_list.npy')
        np.save(r'C:\Users\aiwei\Desktop\codes\points_list.npy', points_list)
        points_list = np.load(r'C:\Users\aiwei\Desktop\codes\points_list.npy')
        L_list = np.zeros(self.ID_max)
        L_list[id_list.astype(int)] = points_list
        plt.plot(L_list)
        plt.show()
        logging.debug(L_list.shape)
        return L_list, L_overlap, temp_id


    @staticmethod
    def gen_good_tie_points(i_j_h_w_points: List[Tuple[ndarray, ndarray]]):
        i_points = []
        j_points = []
        for i_j_h_w_point in i_j_h_w_points:
            # 里面存储的是img_i的列、行，img_j的列、行
            i_point_h_w, j_point_h_w = i_j_h_w_point
            i_w, i_h = i_point_h_w
            j_w, j_h = j_point_h_w
            i_points.append([i_w, i_h])
            j_points.append([j_w, j_h])
        # np.save(fr"C:\Users\aiwei\Desktop\temp\{i}_points.npy", np.array(i_points))
        # np.save(fr"C:\Users\aiwei\Desktop\temp\{j}_points.npy", np.array(j_points))
        # np.save(fr"C:\Users\aiwei\Desktop\temp\radiance_{i}.npy", radiance_i)
        # np.save(fr"C:\Users\aiwei\Desktop\temp\radiance_{j}.npy", radiance_j)
        # i_points = np.load(fr"C:\Users\aiwei\Desktop\temp\{i}_points.npy")
        # j_points = np.load(fr"C:\Users\aiwei\Desktop\temp\{j}_points.npy")
        # radiance_i = np.load(fr"C:\Users\aiwei\Desktop\temp\radiance_{i}.npy")
        # radiance_j = np.load(fr"C:\Users\aiwei\Desktop\temp\radiance_{j}.npy")
        return np.array(i_points), np.array(j_points)

        #     i_points.append(radiance_i[i_h, i_w])
        #     j_points.append(radiance_j[j_h, j_w])
        # i_points_h_w = np.array(i_points)
        # j_points_h_w = np.array(j_points)
        # # 去除奇异值
        # i_points_h_w = utils.filter_ndarray(i_points_h_w)
        # j_points_h_w = utils.filter_ndarray(j_points_h_w)
        # final_i_points_h_w = i_points_h_w[~np.isnan(i_points_h_w) * ~np.isnan(j_points_h_w)]
        # final_j_points_h_w = j_points_h_w[~np.isnan(i_points_h_w) * ~np.isnan(j_points_h_w)]
        #
        # return final_i_points_h_w, final_j_points_h_w


    def gen_tie_points_1(self) -> Tuple[ndarray, ndarray, ndarray]:
        """
        ID_overlap的第i个元素表示overlap图片的ID_i
        L_overlap的第i个元素表示ID_i对应的overlap图片上的一个点的radiance
        E_overlap的第i个元素表示ID_i对应的overlap图片上的一个点的irradiance
        同一个ID_i可以有很多个reference points
        Returns:
            tie points的L
        """
        temp_id = []
        for i in range(self.ID_max):
            for j in range(i + 1, self.ID_max, 1):
                # if (np.abs(i - j) > 3) and ((np.abs(i - j) < 45) or (np.abs(i - j) > 200)):
                #     continue
                if (np.abs(i - j) <= 3) or ((np.abs(i - j) >= 48) and (np.abs(i - j) <= 52))\
                        or ((np.abs(i - j) >= 98) and (np.abs(i - j) <= 102))\
                        or ((np.abs(i - j) >= 148) and (np.abs(i - j) <= 152)):
                # if (np.abs(i - j) <= 3) or ((np.abs(i - j) >= 48) and (np.abs(i - j) <= 52)):
                    img_name1 = f"IMG_{str(i).zfill(4)}_2.tif"
                    img_path1 = os.path.join(self.path['images_dir'], img_name1)
                    img_name2 = f"IMG_{str(j).zfill(4)}_2.tif"
                    img_path2 = os.path.join(self.path['images_dir'], img_name2)
                    if os.path.exists(img_path1) and os.path.exists(img_path2):
                        temp_id.append([i, j])
        temp_id = np.array(temp_id)
        logging.debug(f"{self.all_num}张图，共有{len(temp_id)}种组合")
        L_overlap = np.zeros((self.ID_max, self.ID_max)) * np.nan  # radiance
        id_list, points_list, L_overlap, temp_id = self.f_pool(temp_id, L_overlap)
        np.save(r'C:\Users\aiwei\Desktop\codes\id_list.npy', id_list)
        id_list = np.load(r'C:\Users\aiwei\Desktop\codes\id_list.npy')
        np.save(r'C:\Users\aiwei\Desktop\codes\points_list.npy', points_list)
        points_list = np.load(r'C:\Users\aiwei\Desktop\codes\points_list.npy')
        L_list = np.zeros(self.ID_max)
        L_list[id_list] = points_list
        logging.debug(L_list.shape)
        return L_list, L_overlap, temp_id
        # logging.debug(len(temp_id))
        # with ProcessPoolExecutor(max_workers=6) as executor:
        #     results = list(executor.map(self.f_pool, temp_id))
        #     logging.debug(1)


    def cv_dict(self):
        d1 = typed.Dict.empty(types.int32, types.string)
        for k, v in self.overlap_img_path_dict.items():
            d1[k] = v
        return d1

    @staticmethod
    def get_radiance_from_raw(raw_path: str) -> ndarray:
        """
        获取单张raw图片的radiance数据
        Args:
            raw_path: 原始影像路径，数据是raw
        Returns:
            radiance数据
        """
        raw = plt.imread(raw_path)
        meta = metadata.Metadata(raw_path, exiftoolPath=os.environ.get('exif_tool_path'))
        radiance, _, _, _ = msutils.raw_image_to_radiance(meta, raw)
        return radiance

    def BRDF_fix(self, path: str):
        """
        获取单张raw图片经过BRDF矫正后的数据
        Args:
            path:

        Returns:

        """
        L = self.get_radiance_from_raw(path)
        meta = metadata.Metadata(path, exiftoolPath=os.environ.get('exif_tool_path'))
        meta_all = meta.get_all()
        solar_zenith = (np.pi / 2) - float(meta_all['XMP:SolarElevation'])
        solar_azimuth = float(meta_all['XMP:SolarAzimuth'])
        UAV_height = float(meta_all['Composite:GPSAltitude'])
        h, w = L.shape
        ww, hh = np.meshgrid(np.arange(w), np.arange(h))
        h_, w_ = int(np.floor(h / 2)), int(np.floor(w / 2))  # 中心坐标
        UAV_zenith = (np.pi / 2) - np.arctan(UAV_height / (2.8 * np.sqrt((hh - h_) ** 2 + (ww - w_) ** 2)))
        UAV_azimuth = np.arctan((hh - h_) / (ww - w_))
        cos_phi = np.nanmean(np.cos(UAV_azimuth - solar_azimuth))
        theta_i = solar_zenith
        theta_r = UAV_zenith
        L = L.flatten()
        x = np.linalg.pinv(np.array([(theta_i ** 2) * (theta_r ** 2),
                                     (theta_i ** 2) + (theta_r ** 2),
                                     theta_i * theta_r * cos_phi,
                                     1])) * L
        b1, b2, b3, b4 = x[0], x[1], x[2], x[3]
        return (b1 * (theta_i ** 2) * (theta_r ** 2) + b2 * (theta_i ** 2) + (theta_r ** 2) + b3 * theta_i *
                theta_r * cos_phi + b4) * L





    # @staticmethod
    # @jit
    def f_pool(self, temp_id: ndarray, L_overlap: ndarray):
        len_ = len(temp_id)
        points_list = []
        id_list = []
        for q in tqdm(range(len_)):
            i, j = temp_id[q]
            img_name_i = f"IMG_{str(i).zfill(4)}_2.tif"
            img_path_i = os.path.join(self.path['images_dir'], img_name_i)
            img_name_j = f"IMG_{str(j).zfill(4)}_2.tif"
            img_path_j = os.path.join(self.path['images_dir'], img_name_j)

            # 获取tie points，每一对图像,在其交集区域，都有2个tie point
            max_h1, min_h1, max_w1, min_w1, max_h2, min_h2, max_w2, min_w2 = \
                utils.gen_sift_points(img_path_i, img_path_j)
            if max_h1 != -1:
                img_i = utils.get_radiance_from_raw(img_path_i)  # 图像i的radiance
                img_j = utils.get_radiance_from_raw(img_path_j)  # 图像j的radiance
                # img_i中两个tie point的radiance
                L_max_i, L_min_i = np.log(img_i[max_h1, max_w1]), np.log(img_i[min_h1, min_w1])
                # img_j中两个tie point的radiance
                L_max_j, L_min_j = np.log(img_j[max_h2, max_w2]), np.log(img_i[min_h2, min_w2])
                L_max_diff, L_min_diff = L_max_i - L_max_j, L_min_i - L_min_j
                # 图像i和图像j交集区域的一个tie point的L(radiance)记录
                L_overlap[i, j] = L_max_diff  # 图像i和图像j交集区域的一个tie point的ID记录
                L_overlap[j, i] = L_min_diff  # 图像i和图像j交集区域的一个tie point的ID记录
                # ID_overlap.append([img_i_id, img_j_id])  # 图像i和图像j交集区域的ID记录
                # L_overlap.append([L_max_diff, L_min_diff])  # 图像i和图像j交集区域的一个tie point的ID记录
                id_list.append(i)
                id_list.append(j)
                points_list.append(L_min_i)
                points_list.append(L_min_j)
            else:
                temp_id[q] = -1
        return np.array(id_list), np.array(points_list), L_overlap, temp_id

    def gen_sift_points(self, img_path_i: str, img_path_j: str) -> Tuple[int, int, int, int, int, int, int, int]:
        img_i = cv2.cvtColor(cv2.imread(img_path_i), cv2.COLOR_BGR2GRAY)
        img_j = cv2.cvtColor(cv2.imread(img_path_j), cv2.COLOR_BGR2GRAY)

        # 使用SIFT提取特征点
        sift = cv2.xfeatures2d.SIFT_create()
        # kp1[m1.queryIdx].pt、kp2[m1.trainIdx].pt中存储的是两张图的特征点的坐标(列、行)
        (kp1, des1) = sift.detectAndCompute(img_i, None)
        (kp2, des2) = sift.detectAndCompute(img_j, None)

        # 基于KNN对两张图的特征点进行匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # 根据ratio参数筛选优秀的特征点
        # ratio=0.4：对于准确度要求高的匹配；
        # ratio=0.6：对于匹配点数目要求比较多的匹配；
        # ratio=0.5：一般情况下。
        ratio = 0.4
        good_match_points = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_match_points.append(m)

        # 使用RANSAC进一步筛选点
        if len(good_match_points) > 1:
            # 优秀的特征点的坐标
            ptsA = np.float32([kp1[m.queryIdx].pt for m in good_match_points]).reshape(-1, 1, 2)
            ptsB = np.float32([kp2[m.trainIdx].pt for m in good_match_points]).reshape(-1, 1, 2)
            # RANSAC参数设置
            ransacReprojThreshold = 1
            maxIters = ptsA.shape[0] * (ptsA.shape[0] - 1)
            _, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold, maxIters=maxIters)
            # 最优点的坐标
            temp_h1 = np.round(ptsA[status[:, 0] == 1, 0, ::-1][:, 0]).astype(int)
            temp_w1 = np.round(ptsA[status[:, 0] == 1, 0, ::-1][:, 1]).astype(int)
            temp_h2 = np.round(ptsB[status[:, 0] == 1, 0, ::-1][:, 0]).astype(int)
            temp_w2 = np.round(ptsB[status[:, 0] == 1, 0, ::-1][:, 1]).astype(int)
            # 去除最优点中的奇异值
            img_i_filter = utils.filter_ndarray(img_i[temp_h1, temp_w1])
            img_j_filter = utils.filter_ndarray(img_j[temp_h2, temp_w2])
            # 获取最优点中DN值最大和最小的两个点
            max_ = np.where((img_i_filter == np.nanmax(img_i_filter[~np.isnan(img_j_filter)])))[0][0]
            min_ = np.where((img_i_filter == np.nanmin(img_i_filter[~np.isnan(img_j_filter)])))[0][0]
            max_h1, min_h1, max_w1, min_w1 = temp_h1[max_], temp_h1[min_], temp_w1[max_], temp_w1[min_]
            max_h2, min_h2, max_w2, min_w2 = temp_h2[max_], temp_h2[min_], temp_w2[max_], temp_w2[min_]
            return max_h1, min_h1, max_w1, min_w1, max_h2, min_h2, max_w2, min_w2
        else:
            return -1, -1, -1, -1, -1, -1, -1, -1

    def gen_ref_type_json_num(self) -> dict:
        """
        将板子的颜色设置编号
        Returns:
            编号关于板子颜色名字的字典
        """
        ref_type_json_num = {}
        for i in range(len(self.ref_type)):
            ref_type_json_num[self.ref_type[i]] = i
        return ref_type_json_num

    def from_reference_ID_to_SET_num(self, ID: int) -> str or None:
        """
        根据reference ID获取其所在的SET
        Args:
            ID: reference图片ID
        Returns:
            SET的名字
        """
        for set_num in self.SET_.keys():
            if ID in self.SET_[set_num]:
                return set_num
        return None

    def get_reference_points(self) -> Tuple[ndarray, ndarray, ndarray]:
    # def get_reference_points(self) -> Tuple[ndarray, ndarray, ndarray, list]:
        """
        ID_reference的第i个元素表示reference图片的ID_i
        L_reference的第i个元素表示ID_i对应的reference图片上的一个点的radiance
        E_reference的第i个元素表示ID_i对应的reference图片上的一个点的irradiance
        同一个ID_i可以有很多个reference points
        Returns:
            reference points的L、R
        """
        #
        # E_reference = []  # irradiance
        L_reference = []  # radiance
        R_reference = []  # reflectance factor
        ID_reference = []
        for ID in self.ref_id:
            ref_img_path = self.ref_img_path_dict[ID]
            ref_json_path = self.ref_json_path_dict[ID]
            try:
                set_num = self.from_reference_ID_to_SET_num(ID)
            except TypeError:
                continue
            for ref_type in self.ref_type:
                ID_reference.append(ID)
                try:
                    R_reference.append(self.SET_REF[set_num][self.band][ref_type])
                except KeyError:
                    continue
                logging.debug(fr"编号为{ID}的reference图片，路径为{ref_img_path}， json路径为{ref_json_path}，属于{set_num}"
                              f"，选择的反射率板子的颜色为{ref_type}，{self.band}的反射率为{R_reference[-1]}")
                img = self.get_radiance_from_raw(ref_img_path)  # 获取reference图片的radiance值
                try:
                    ref_mask = self.gen_mask(img, ref_json_path, ref_type)  # 根据json获取mask
                except KeyError:
                    continue
                L_reference.append(np.nanmean(utils.filter_ndarray(img * ref_mask)))
                # E_reference.append(L_reference[-1] * np.pi / R_reference[-1])
        L_reference = np.log(L_reference)
        # E_reference = np.log(E_reference)
        R_reference = np.log(R_reference)
        ID_reference = np.array(ID_reference)  # ID
        # return L_reference, E_reference, R_reference, ID_reference
        return L_reference, R_reference, ID_reference

    def get_overlap_img_path(self) -> Tuple[dict, list]:
        """
        ID_list中保存了全部overlap图片的ID
        path_dict是所有overlap的图片路径关于ID的字典
        Returns:
            获取overlap图像的D的列表，图像路径跟随ID的字典和
        """
        path_dict = {}
        ID_list = []
        for img_name in os.listdir(self.images_dir):
            img_path = os.path.join(self.images_dir, img_name)
            # if img_path in self.ref_img_path_dict.values():
            #     continue
            # else:
            ID = int(img_name.split('_')[1])
            ID_list.append(ID)
            path_dict[ID] = img_path
        return path_dict, ID_list

    def get_reference_img_path(self) -> Tuple[dict, dict, list]:
        """
        ID_list中保存了全部reference图片的ID
        path_dict是所有reference的图片路径关于ID的字典
        json_path_dict是所有reference的json路径关于ID的字典
        Returns:
            获取reference图像的D的列表，图像路径跟随ID的字典和对应的json路径跟随ID的字典
        """
        path_dict = {}
        json_path_dict = {}
        ID_list = []
        reference_json_root_path = self.path['ref_json']
        for set_num in self.SET_.keys():
            for ID in self.SET_[f'{set_num}']:
                reference_img_name = fr'IMG_{str(ID).zfill(4)}_2.tif'
                reference_json_name = fr'IMG_{str(ID).zfill(4)}_1.json'
                ID_list.append(ID)
                path_dict[ID] = os.path.join(self.images_dir, reference_img_name)
                json_path_dict[ID] = os.path.join(reference_json_root_path, reference_json_name)
        return path_dict, json_path_dict, ID_list

    def gen_SET_REF_dict(self) -> dict:
        """
        SET_REF[所属集合编号][波段号][板子颜色] = 对应的板子的反射率
        Returns:
            不同波段，不同SET，不同颜色的板子的反射率字典
        """
        SET_REF = {}
        df = pd.read_csv(self.path['target_ref'], index_col=0)
        for set_num in self.SET_.keys():
            SET_REF[set_num] = {}
            for band in df.index.values:
                SET_REF[set_num][band] = {}
                for type_ in self.ref_type:
                    SET_REF[set_num][band][type_] = float(df.loc[band][f'{set_num}_{type_}'])
        return SET_REF

    def micasense_DLS(self, images_dir: str) -> pd.DataFrame:
        """
        利用micasense获取整个图像集合的DLS数据
        Args:
            images_dir: 图像集合的路径
        Returns:
            DataFrame对象，其中index是拍摄时间，columns为每张图片的三个DLS姿态、不同波段的irradiance
        """
        imgset = imageset.ImageSet.from_directory(images_dir, exiftool_path=self.exif_tool_Path)
        data, columns = imgset.as_nested_lists()
        df = pd.DataFrame.from_records(data, index='timestamp', columns=columns)
        # logging.debug(df)
        return df

    def gen_mask(self, img: ndarray, json_path: str, ref_type: str) -> ndarray:
        """
        根据json文件生成图片矩阵的mask,
        Args:
            img: 图片数据矩阵
            json_path: 使用labelme对img进行标注之后存储的json文件的路径
            ref_type: 反射板的颜色
        Returns:
            与img的shape一致的mask矩阵，根据json文件，指定类别（ref_type）的内部数值为1，其它区域为nan
        """
        labelme_json = json.load(open(json_path, encoding='utf-8'))
        reference_radiance = np.zeros(img.shape) * np.nan
        mask = np.zeros(reference_radiance.shape) * np.nan
        points = labelme_json['shapes'][self.ref_type_json_num[ref_type]]['points']  # white
        points = np.array(points).astype(np.int32)
        cv2.fillConvexPoly(mask, points, 1)
        return mask