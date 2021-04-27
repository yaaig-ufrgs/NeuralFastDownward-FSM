(define (problem roverprob7126) (:domain rover)
(:objects
	general - lander
	colour high_res low_res - mode
	rover0 rover1 rover2 rover3 - rover
	rover0store rover1store rover2store rover3store - store
	waypoint0 waypoint1 waypoint2 waypoint3 waypoint4 waypoint5 waypoint6 waypoint7 - waypoint
	camera0 camera1 camera2 camera3 - camera
	objective0 objective1 objective2 - objective
	)
		(:init
		(at rover0 waypoint0)
		(at rover1 waypoint4)
		(at rover2 waypoint7)
		(at rover3 waypoint2)
		(at_rock_sample waypoint4)
		(communicated_image_data objective1 high_res)
		(communicated_rock_data waypoint7)
		(communicated_soil_data waypoint0)
		(communicated_soil_data waypoint1)
		(communicated_soil_data waypoint3)
		(communicated_soil_data waypoint6)
		(empty rover0store)
		(empty rover2store)
		(empty rover3store)
		(full rover1store)
		(have_image rover1 objective1 high_res)
		(have_image rover3 objective1 high_res)
		(have_rock_analysis rover1 waypoint0)
		(have_rock_analysis rover1 waypoint1)
		(have_rock_analysis rover1 waypoint2)
		(have_rock_analysis rover1 waypoint6)
		(have_rock_analysis rover1 waypoint7)
		(have_soil_analysis rover1 waypoint0)
		(have_soil_analysis rover2 waypoint3)
		(have_soil_analysis rover2 waypoint6)
		(have_soil_analysis rover3 waypoint1)
		(visible waypoint0 waypoint4)
		(visible waypoint4 waypoint0)
		(visible waypoint0 waypoint7)
		(visible waypoint7 waypoint0)
		(visible waypoint1 waypoint0)
		(visible waypoint0 waypoint1)
		(visible waypoint1 waypoint5)
		(visible waypoint5 waypoint1)
		(visible waypoint1 waypoint7)
		(visible waypoint7 waypoint1)
		(visible waypoint2 waypoint1)
		(visible waypoint1 waypoint2)
		(visible waypoint3 waypoint0)
		(visible waypoint0 waypoint3)
		(visible waypoint3 waypoint1)
		(visible waypoint1 waypoint3)
		(visible waypoint3 waypoint2)
		(visible waypoint2 waypoint3)
		(visible waypoint3 waypoint6)
		(visible waypoint6 waypoint3)
		(visible waypoint4 waypoint1)
		(visible waypoint1 waypoint4)
		(visible waypoint4 waypoint5)
		(visible waypoint5 waypoint4)
		(visible waypoint5 waypoint2)
		(visible waypoint2 waypoint5)
		(visible waypoint5 waypoint3)
		(visible waypoint3 waypoint5)
		(visible waypoint5 waypoint6)
		(visible waypoint6 waypoint5)
		(visible waypoint6 waypoint0)
		(visible waypoint0 waypoint6)
		(visible waypoint6 waypoint2)
		(visible waypoint2 waypoint6)
		(visible waypoint6 waypoint4)
		(visible waypoint4 waypoint6)
		(visible waypoint7 waypoint2)
		(visible waypoint2 waypoint7)
		(visible waypoint7 waypoint3)
		(visible waypoint3 waypoint7)
		(visible waypoint7 waypoint5)
		(visible waypoint5 waypoint7)
		(at_lander general waypoint1)
		(channel_free general)
		(available rover0)
		(store_of rover0store rover0)
		(equipped_for_soil_analysis rover0)
		(equipped_for_imaging rover0)
		(can_traverse rover0 waypoint1 waypoint0)
		(can_traverse rover0 waypoint0 waypoint1)
		(can_traverse rover0 waypoint1 waypoint3)
		(can_traverse rover0 waypoint3 waypoint1)
		(can_traverse rover0 waypoint1 waypoint4)
		(can_traverse rover0 waypoint4 waypoint1)
		(can_traverse rover0 waypoint1 waypoint5)
		(can_traverse rover0 waypoint5 waypoint1)
		(can_traverse rover0 waypoint0 waypoint6)
		(can_traverse rover0 waypoint6 waypoint0)
		(can_traverse rover0 waypoint0 waypoint7)
		(can_traverse rover0 waypoint7 waypoint0)
		(can_traverse rover0 waypoint3 waypoint2)
		(can_traverse rover0 waypoint2 waypoint3)
		(available rover1)
		(store_of rover1store rover1)
		(equipped_for_soil_analysis rover1)
		(equipped_for_rock_analysis rover1)
		(equipped_for_imaging rover1)
		(can_traverse rover1 waypoint3 waypoint0)
		(can_traverse rover1 waypoint0 waypoint3)
		(can_traverse rover1 waypoint3 waypoint7)
		(can_traverse rover1 waypoint7 waypoint3)
		(can_traverse rover1 waypoint0 waypoint6)
		(can_traverse rover1 waypoint6 waypoint0)
		(can_traverse rover1 waypoint7 waypoint1)
		(can_traverse rover1 waypoint1 waypoint7)
		(can_traverse rover1 waypoint7 waypoint2)
		(can_traverse rover1 waypoint2 waypoint7)
		(can_traverse rover1 waypoint6 waypoint4)
		(can_traverse rover1 waypoint4 waypoint6)
		(can_traverse rover1 waypoint6 waypoint5)
		(can_traverse rover1 waypoint5 waypoint6)
		(available rover2)
		(store_of rover2store rover2)
		(equipped_for_soil_analysis rover2)
		(equipped_for_imaging rover2)
		(can_traverse rover2 waypoint3 waypoint0)
		(can_traverse rover2 waypoint0 waypoint3)
		(can_traverse rover2 waypoint3 waypoint1)
		(can_traverse rover2 waypoint1 waypoint3)
		(can_traverse rover2 waypoint3 waypoint2)
		(can_traverse rover2 waypoint2 waypoint3)
		(can_traverse rover2 waypoint3 waypoint5)
		(can_traverse rover2 waypoint5 waypoint3)
		(can_traverse rover2 waypoint3 waypoint6)
		(can_traverse rover2 waypoint6 waypoint3)
		(can_traverse rover2 waypoint3 waypoint7)
		(can_traverse rover2 waypoint7 waypoint3)
		(can_traverse rover2 waypoint0 waypoint4)
		(can_traverse rover2 waypoint4 waypoint0)
		(available rover3)
		(store_of rover3store rover3)
		(equipped_for_soil_analysis rover3)
		(equipped_for_imaging rover3)
		(can_traverse rover3 waypoint7 waypoint0)
		(can_traverse rover3 waypoint0 waypoint7)
		(can_traverse rover3 waypoint7 waypoint1)
		(can_traverse rover3 waypoint1 waypoint7)
		(can_traverse rover3 waypoint7 waypoint3)
		(can_traverse rover3 waypoint3 waypoint7)
		(can_traverse rover3 waypoint7 waypoint5)
		(can_traverse rover3 waypoint5 waypoint7)
		(can_traverse rover3 waypoint0 waypoint4)
		(can_traverse rover3 waypoint4 waypoint0)
		(can_traverse rover3 waypoint1 waypoint2)
		(can_traverse rover3 waypoint2 waypoint1)
		(can_traverse rover3 waypoint3 waypoint6)
		(can_traverse rover3 waypoint6 waypoint3)
		(on_board camera0 rover1)
		(calibration_target camera0 objective1)
		(supports camera0 high_res)
		(supports camera0 low_res)
		(on_board camera1 rover2)
		(calibration_target camera1 objective0)
		(supports camera1 colour)
		(supports camera1 high_res)
		(on_board camera2 rover3)
		(calibration_target camera2 objective0)
		(supports camera2 high_res)
		(on_board camera3 rover0)
		(calibration_target camera3 objective1)
		(supports camera3 colour)
		(visible_from objective0 waypoint0)
		(visible_from objective0 waypoint1)
		(visible_from objective0 waypoint2)
		(visible_from objective0 waypoint3)
		(visible_from objective0 waypoint4)
		(visible_from objective0 waypoint5)
		(visible_from objective1 waypoint0)
		(visible_from objective1 waypoint1)
		(visible_from objective1 waypoint2)
		(visible_from objective2 waypoint0)
		(visible_from objective2 waypoint1)
		(visible_from objective2 waypoint2)
		(visible_from objective2 waypoint3)
		(visible_from objective2 waypoint4)
		(visible_from objective2 waypoint5)
		(visible_from objective2 waypoint6)
		)

(:goal (and
(communicated_soil_data waypoint6)
(communicated_soil_data waypoint0)
(communicated_soil_data waypoint1)
(communicated_soil_data waypoint3)
(communicated_rock_data waypoint6)
(communicated_rock_data waypoint0)
(communicated_rock_data waypoint4)
(communicated_rock_data waypoint7)
(communicated_image_data objective1 high_res)
	)
)
)

