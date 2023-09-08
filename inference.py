import cv2 as cv
import numpy as np
import tensorflow as tf



def same_row(row, winner_type):
    if winner_type in [0, 1]:
        return row[0] == row[1] == row[2] == winner_type


def find_the_winner(pred):
    types = pred[:, -3:].argmax(axis=1).reshape((3, 3))
    pred_tmp = pred.reshape((3, 3, 7))
    
    for i in range(3):
        if same_row(types[:, i], types[0, i]):
            pred_tmp = pred_tmp[:, i]
            line_c = np.array([(pred_tmp[0, 0] + pred_tmp[0, 2]) / 2, pred_tmp[0, 1], (pred_tmp[2, 0] + pred_tmp[2, 2]) / 2, pred_tmp[2, 3], types[0, i]])
            return line_c
        
        if same_row(types[i, :], types[i, 0]):
            pred_tmp = pred_tmp[i, :]
            line_c = np.array([pred_tmp[0, 0], (pred_tmp[0, 1] + pred_tmp[0, 3]) / 2, pred_tmp[2, 2], (pred_tmp[2, 1] + pred_tmp[2, 3]) / 2, types[i, 0]])
            return line_c
        
    if same_row([types[i, i] for i in range(3)], types[0, 0]):
        pred_tmp = np.array([pred_tmp[i, i] for i in range(3)])
        line_c = np.array([pred_tmp[0, 0], pred_tmp[0, 1], pred_tmp[2, 2], pred_tmp[2, 3], types[0, 0]])
        return line_c
    
    if same_row([types[i, 2 - i] for i in range(3)], types[0, 2]):
        pred_tmp = np.array([pred_tmp[i, 2 - i] for i in range(3)])
        line_c = np.array([pred_tmp[0, 2], pred_tmp[0, 1], pred_tmp[2, 0], pred_tmp[2, 3],  types[0, 2]])
        return line_c


def write_new_image(input_path, output_path):
	model_tic_tac_toe = tf.keras.models.load_model("models/model_tic_tac_toe.h5")
	
	image_path = input_path
	color_dict = {0: (0, 150, 0), 1: (0, 0, 200), 2: (100, 100, 100)}
	winner_dict = {0: "O", 1: "X"}


	img = cv.imread(image_path)
	img_pred_tmp = cv.cvtColor(cv.resize(img, (150, 150), interpolation=cv.INTER_AREA), cv.COLOR_BGR2GRAY)
	    
	pred = np.array(model_tic_tac_toe.predict(img_pred_tmp[None] / 255)).reshape((9, 7))
    
	winner = find_the_winner(pred)
	   
	if winner is not None:
		cv.line(img, (int(winner[0] * img.shape[1]), int(winner[1] * img.shape[0])), (int(winner[2] * img.shape[1]), int(winner[3] * img.shape[0])), color_dict[winner[-1]], thickness=5)
	    
	    
	cv.imwrite(output_path, img)




if __name__ == "__main__":
	input_path = input("\n\nWrite a path of the input file (example init_data/images/tic_{number 01-12}.jpg): ")
	output_path = input("\n\nWrite a path of the output file: ")
	write_new_image(input_path, output_path)
