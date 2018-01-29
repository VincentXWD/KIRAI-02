import multiprocessing as mp
import path_parser
import os
import padding_process


def mkdir(path: str) -> None:
  path = path.strip()
  path = path.rstrip("\\")
  isExists = os.path.exists(path)
  if not isExists:
    os.makedirs(path)


def rbd_realize(pre_pred_png: list, pre_raw_jpg, tmp_result_png: list) -> list:
  '''
  断点续传
  :param pre_pred_png:
  :param tmp_result_png:
  :return:
  '''
  cnt = 0
  pred_png, raw_jpg = [], []
  result_png = []
  for i in range(0, len(tmp_result_png)):
    for j in range(0, len(tmp_result_png[i][0])):
      result_png.append(tmp_result_png[i][0][j][1])

  for i in range(0, len(pre_pred_png)):
    pred_png.append([[], pre_pred_png[i][1]])
    raw_jpg.append([[], pre_raw_jpg[i][1]])

    for j in range(0, len(pre_pred_png[i][0])):
      flag = 1
      for path in result_png:
        if path == pre_pred_png[i][0][j][1]:
          flag = 0
          break
      if flag == 1:
        pred_png[i][0].append([pre_pred_png[i][0][j][0], pre_pred_png[i][0][j][1]])
        raw_jpg[i][0].append([pre_raw_jpg[i][0][j][0], pre_raw_jpg[i][0][j][1]])
        cnt += 1

  print('have', cnt, ' files left.')
  return pred_png, raw_jpg


def single_process(each: list) -> None:
  raw_jpg = each[0]
  pred_png = each[1]
  save_path = each[2]

  for i in range(0, len(pred_png)):
    pred_dir_name = pred_png[i][1]
    pred = pred_png[i]
    raw = raw_jpg[i]

    mkdir(save_path + pred_dir_name)

    for j in range(0, len(pred[0])):
      pred_path, pred_name = pred[0][j][0], pred[0][j][1]
      raw_path, raw_name = raw[0][j][0], raw[0][j][1]
      save_name = save_path+pred_dir_name+'/'+pred_name
      print('[Multi Processing]'+pred_path, raw_path, save_name)
      padding_process.padding_process(raw_path, pred_path, save_name)
      print('[Multi Processing] saved :' + save_name)


def multi_processing_sta1():
  # [i][0][j][0] : 目录路径 ; [i][1] : 目录名
  # [i][0][j][1] : 切片名
  # pred_path = './images/pred_png/'
  # save_path = './padded/'
  # raw_path = './images/raw_jpg/'
  pred_path = './processing/model_output/'
  save_path = './processing/padded/'
  raw_path = './processing/raw_image/'

  mkdir(save_path)

  pre_raw_jpg = path_parser.get_file_path(raw_path)
  pre_pred_png = path_parser.get_file_path(pred_path)
  tmp_result_png = path_parser.get_file_path(save_path)
  pred_png, raw_jpg = rbd_realize(pre_pred_png, pre_raw_jpg, tmp_result_png)

  process_count = 1
  each_process = []
  i, j = 0, 0
  while j <= len(pred_png):
    j = min(i + process_count, len(pred_png))
    each_process.append([raw_jpg[i:j], pred_png[i:j], save_path])
    i = j
    j += process_count

  for each in each_process:
    p = mp.Process(target=single_process, args=(each,))
    p.start()


def jumping_board(each: list) -> None:
  raw_jpg = each[0]
  pred_png = each[1]
  save_path = each[2]
  padding_process.padding_process(raw_jpg, pred_png, save_path)


def multi_processing_sta2():
  # [i][0][j][0] : 目录路径 ; [i][1] : 目录名
  # [i][0][j][1] : 切片名
  pred_path = './images/pred_png/'
  save_path = './padded/'
  raw_path = './images/raw_jpg/'
  mkdir(save_path)

  pre_raw_jpg = path_parser.get_file_path(raw_path)
  pre_pred_png = path_parser.get_file_path(pred_path)
  tmp_result_png = path_parser.get_file_path(save_path)
  pred_png, raw_jpg = rbd_realize(pre_pred_png, pre_raw_jpg, tmp_result_png)

  process_count = mp.cpu_count()

  print('[Multi Processing] Task number:', process_count)
  flat_task = []

  print('[Multi Processing] Now distributing tasks...')

  for i in range(0, len(pred_png)):
    pred_dir_name = pred_png[i][1]
    pred = pred_png[i]
    raw = raw_jpg[i]
    mkdir(save_path + pred_dir_name)
    for j in range(0, len(pred_png[i][0])):
      pred_path, pred_name = pred[0][j][0], pred[0][j][1]
      raw_path, raw_name = raw[0][j][0], raw[0][j][1]

      save_name = save_path+pred_dir_name+'/'+pred_name
      flat_task.append([raw_path, pred_path, save_name])

  print('[Multi Processing] Task distributed.')
  print('[Multi Processing] Now processing...')

  pool = mp.Pool(processes=process_count)

  for i in range(0, len(flat_task)):
    pool.apply_async(jumping_board, (flat_task[i], ))

  print('[Multi Processing] Process pool closed.')
  pool.close()
  pool.join()
  print('[Multi Processing] Finished.')

if __name__ == '__main__':
  multi_processing_sta2()
