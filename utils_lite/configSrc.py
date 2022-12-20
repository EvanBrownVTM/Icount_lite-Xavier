##### NETWORK #####
IP_ADDRESS_LOCAL="192.168.1.155"
IP_ADDRESS_PI="192.168.1.65"
#IP_ADDRESS_PI="localhost"

##### Device specific #####
user = 'cvnx'
log_path = "/home/{}/Desktop/".format(user)
base_path = "/home/{}/Desktop/Icount_lite/".format(user)

##### Location specific #####
camera_map =  {"cam0": 23881566, "cam1":23881567, "cam2":23875565, "cam3":23897010}
activate_arch = True
sms_alert = False
#cls_dict = {0: 'vickies', 1: 'Coca Cola 20z', 2: 'Gatorade 12oz', 3: 'sun_chips', 4: 'doritos', 5: 'lays', 6: 'monster', 7: 'gold_peak', 8: 'Diet Coke (SF)', 9: 'sprite'} #10 prod
cls_dict = {0: 'Coca Cola 20z', 1: 'Gatorade 12oz', 2: 'Dasani Water 16 oz', 3: 'Monster Regular'} #drinks

cam0_zone = 'utils_lite/regus_lab_cam0_nt.npz'
cam1_zone = 'utils_lite/regus_lab_cam1_nt.npz'
cam2_zone = 'utils_lite/regus_live_cam2.npz'

##### Software setting #####
archive_flag = True
maxCamerasToUse = 3
archive_size = 416
save_size = 200
display_mode = False
pika_flag = True
