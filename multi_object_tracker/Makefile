CPP = g++

# OpenCV 3.4.0
#CPPFLAGS = -L/home/sathish/installations/opencv-3.4.0/install/lib \
	     -I/home/sathish/installations/opencv-3.4.0/install/include \
	     -I/home/sathish/installations/opencv-3.4.0/install/include/opencv \
	     -std=c++11 \
	     -g

DEBUG_LEVEL     = -g	\
				  -std=c++11
# Opencv 3.4.3
CPPFLAGS = -I/home/sathish/installations/cv-3.4.3/opencv-3.4.3/install/include/opencv	\
		   -I/home/sathish/installations/cv-3.4.3/opencv-3.4.3/install/include
	       

LDFLAGS = -L/home/sathish/installations/cv-3.4.3/opencv-3.4.3/install/lib
#LDLIBS = -lopencv_superres -lopencv_stitching -lopencv_videostab -lopencv_line_descriptor -lopencv_hfs -lopencv_dpm	\
		-lopencv_structured_light -lopencv_img_hash -lopencv_bioinspired -lopencv_sfm -lopencv_rgbd -lopencv_xphoto	\
		-lopencv_freetype -lopencv_xobjdetect -lopencv_hdf -lopencv_phase_unwrapping -lopencv_ccalib -lopencv_aruco	\
		-lopencv_reg -lopencv_cvv -lopencv_surface_matching -lopencv_bgsegm -lopencv_fuzzy -lopencv_saliency	\
		-lopencv_dnn_objdetect -lopencv_face -lopencv_photo -lopencv_objdetect -lopencv_tracking -lopencv_datasets	\
		-lopencv_text -lopencv_dnn -lopencv_plot -lopencv_stereo -lopencv_xfeatures2d -lopencv_shape -lopencv_ml	\
		-lopencv_optflow -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio	\
		-lopencv_imgcodecs -lopencv_video -lopencv_imgproc -lopencv_flann -lopencv_core
		#-ldl -lm -lpthread -lrt -L/usr/lib/x86_64-linux-gnu -lGLU -lGL

LDLIBS = -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_cvv -lopencv_dpm -lopencv_face -lopencv_photo -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_sfm -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ml -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_flann -lopencv_xobjdetect -lopencv_imgcodecs -lopencv_objdetect -lopencv_xphoto -lopencv_imgproc -lopencv_core		

all: test

test: multi_tracker.cpp
	$(CPP) $(DEBUG_LEVEL) $^ -o $@ $(CPPFLAGS) $(LDFLAGS) $(LDLIBS) 
