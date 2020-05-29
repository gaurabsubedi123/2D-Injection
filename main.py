import os
import cv2
import time
import lib
lib.launch()


advert_dir = "input_advert" #insert input image directory path
input_dir  = "input_videos" #insert input video directory path
output_dir = "output_videos" #insert output video directory path i.e directory where video is to be saved.


"GOOD VIDEO DATASET PATHS"
path1a = os.path.join(input_dir, 'test.mp4') #insert the test video path


"ADVERT IMAGE DATASET PATHS"
path2a = os.path.join(advert_dir, 'input_advert/ad.jpg') #insert image path

def EAR_TEST_CASE1(path1=path1a, path2=path2a, path3=output_dir):
  print("\n=======================INITIALIZATION==========================")
  Frames = lib.load_video(path1, volume=9999)
  advert = lib.load_advert(path2)
  advert = lib.advert_modify(advert, mag=1, space=5)
  origin_points   = lib.get_corners(advert)
  marker_points   = lib.planar_region(Frames[0], method='smart', show=False, scale=0.8, window_size=(50,50), hstride=10, vstride=10)
  fWidth, fHeight = Frames[0].shape[:-1][::-1]
  save_video = cv2.VideoWriter(os.path.join(path3, 'Output.mp4'), 
   cv2.VideoWriter_fourcc(*'MP4V'), 30, (fWidth, fHeight)) #insert output video path with output video name
  print("\n============PERFORM ENHANCED AUGMENTED REALITY================")

  Tic = time.time()
  for i in range(len(Frames)-1):
          src_pts, dst_pts = lib.match_frames(Frames[i], Frames[i+1], method='BFMBased', num_feats=1800, num_select=0.5)
          HH1, mask1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0); HH1 = HH1/HH1[2,2]
          marker_points = cv2.perspectiveTransform(marker_points, HH1)
          HH2 = cv2.getPerspectiveTransform(origin_points, marker_points); HH2 = HH2/HH2[2,2]
          warped_frame = cv2.warpPerspective(advert, HH2, Frames[i+1].shape[:-1][::-1], Frames[i+1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
          save_video.write(warped_frame)
  Toc = time.time()
  save_video.release()

  print("\n========================COMPLETE=============================")
  print("Total Frames  : ", len(Frames))
  print("Time Elapsed  : ", (Toc - Tic))
  print("Estimated FPS : ", len(Frames) / float((Toc - Tic)))
  del Frames


if __name__ == '__main__':
  EAR_TEST_CASE1(path1a, path2a)#ad in vd
