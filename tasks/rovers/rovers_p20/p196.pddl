(define (problem roverprob7182) (:domain rover)
(:objects
	general - lander
	colour high_res low_res - mode
	rover0 rover1 rover2 rover3 rover4 rover5 rover6 rover7 - rover
	rover0store rover1store rover2store rover3store rover4store rover5store rover6store rover7store - store
	waypoint0 waypoint1 waypoint2 waypoint3 waypoint4 waypoint5 waypoint6 waypoint7 waypoint8 waypoint9 waypoint10 waypoint11 waypoint12 waypoint13 waypoint14 waypoint15 waypoint16 waypoint17 waypoint18 waypoint19 waypoint20 waypoint21 waypoint22 waypoint23 waypoint24 - waypoint
	camera0 camera1 camera2 camera3 camera4 camera5 camera6 - camera
	objective0 objective1 objective2 objective3 objective4 objective5 objective6 objective7 - objective
	)
		(:init
		(at rover0 waypoint18)
		(at rover1 waypoint21)
		(at rover2 waypoint3)
		(at rover3 waypoint22)
		(at rover4 waypoint0)
		(at rover5 waypoint18)
		(at rover6 waypoint9)
		(at rover7 waypoint0)
		(at_rock_sample waypoint12)
		(at_rock_sample waypoint14)
		(at_rock_sample waypoint16)
		(at_rock_sample waypoint21)
		(at_soil_sample waypoint1)
		(at_soil_sample waypoint13)
		(at_soil_sample waypoint17)
		(at_soil_sample waypoint20)
		(at_soil_sample waypoint22)
		(at_soil_sample waypoint23)
		(at_soil_sample waypoint24)
		(at_soil_sample waypoint7)
		(at_soil_sample waypoint8)
		(at_soil_sample waypoint9)
		(calibrated camera4 rover1)
		(communicated_image_data objective2 high_res)
		(communicated_image_data objective3 colour)
		(communicated_image_data objective5 high_res)
		(communicated_rock_data waypoint10)
		(communicated_rock_data waypoint18)
		(empty rover0store)
		(empty rover2store)
		(empty rover4store)
		(empty rover6store)
		(empty rover7store)
		(full rover3store)
		(have_image rover1 objective2 high_res)
		(have_image rover1 objective5 high_res)
		(have_image rover4 objective2 high_res)
		(have_image rover4 objective3 colour)
		(have_image rover5 objective5 high_res)
		(have_image rover6 objective3 colour)
		(have_rock_analysis rover3 waypoint18)
		(have_rock_analysis rover3 waypoint22)
		(have_rock_analysis rover3 waypoint9)
		(have_rock_analysis rover4 waypoint10)
		(have_rock_analysis rover4 waypoint7)
		(have_soil_analysis rover0 waypoint11)
		(have_soil_analysis rover0 waypoint18)
		(have_soil_analysis rover3 waypoint0)
		(have_soil_analysis rover6 waypoint15)
		(visible waypoint0 waypoint5)
		(visible waypoint5 waypoint0)
		(visible waypoint0 waypoint15)
		(visible waypoint15 waypoint0)
		(visible waypoint0 waypoint23)
		(visible waypoint23 waypoint0)
		(visible waypoint1 waypoint23)
		(visible waypoint23 waypoint1)
		(visible waypoint2 waypoint1)
		(visible waypoint1 waypoint2)
		(visible waypoint2 waypoint5)
		(visible waypoint5 waypoint2)
		(visible waypoint2 waypoint6)
		(visible waypoint6 waypoint2)
		(visible waypoint2 waypoint14)
		(visible waypoint14 waypoint2)
		(visible waypoint2 waypoint16)
		(visible waypoint16 waypoint2)
		(visible waypoint2 waypoint23)
		(visible waypoint23 waypoint2)
		(visible waypoint3 waypoint1)
		(visible waypoint1 waypoint3)
		(visible waypoint3 waypoint13)
		(visible waypoint13 waypoint3)
		(visible waypoint3 waypoint15)
		(visible waypoint15 waypoint3)
		(visible waypoint3 waypoint18)
		(visible waypoint18 waypoint3)
		(visible waypoint3 waypoint20)
		(visible waypoint20 waypoint3)
		(visible waypoint4 waypoint1)
		(visible waypoint1 waypoint4)
		(visible waypoint4 waypoint6)
		(visible waypoint6 waypoint4)
		(visible waypoint4 waypoint13)
		(visible waypoint13 waypoint4)
		(visible waypoint4 waypoint16)
		(visible waypoint16 waypoint4)
		(visible waypoint4 waypoint19)
		(visible waypoint19 waypoint4)
		(visible waypoint4 waypoint20)
		(visible waypoint20 waypoint4)
		(visible waypoint5 waypoint1)
		(visible waypoint1 waypoint5)
		(visible waypoint5 waypoint4)
		(visible waypoint4 waypoint5)
		(visible waypoint5 waypoint8)
		(visible waypoint8 waypoint5)
		(visible waypoint5 waypoint16)
		(visible waypoint16 waypoint5)
		(visible waypoint6 waypoint5)
		(visible waypoint5 waypoint6)
		(visible waypoint6 waypoint8)
		(visible waypoint8 waypoint6)
		(visible waypoint6 waypoint9)
		(visible waypoint9 waypoint6)
		(visible waypoint6 waypoint11)
		(visible waypoint11 waypoint6)
		(visible waypoint6 waypoint21)
		(visible waypoint21 waypoint6)
		(visible waypoint7 waypoint0)
		(visible waypoint0 waypoint7)
		(visible waypoint7 waypoint4)
		(visible waypoint4 waypoint7)
		(visible waypoint7 waypoint8)
		(visible waypoint8 waypoint7)
		(visible waypoint7 waypoint16)
		(visible waypoint16 waypoint7)
		(visible waypoint7 waypoint17)
		(visible waypoint17 waypoint7)
		(visible waypoint8 waypoint4)
		(visible waypoint4 waypoint8)
		(visible waypoint8 waypoint16)
		(visible waypoint16 waypoint8)
		(visible waypoint8 waypoint17)
		(visible waypoint17 waypoint8)
		(visible waypoint9 waypoint2)
		(visible waypoint2 waypoint9)
		(visible waypoint9 waypoint5)
		(visible waypoint5 waypoint9)
		(visible waypoint9 waypoint7)
		(visible waypoint7 waypoint9)
		(visible waypoint9 waypoint14)
		(visible waypoint14 waypoint9)
		(visible waypoint9 waypoint17)
		(visible waypoint17 waypoint9)
		(visible waypoint10 waypoint11)
		(visible waypoint11 waypoint10)
		(visible waypoint10 waypoint15)
		(visible waypoint15 waypoint10)
		(visible waypoint11 waypoint8)
		(visible waypoint8 waypoint11)
		(visible waypoint11 waypoint19)
		(visible waypoint19 waypoint11)
		(visible waypoint12 waypoint3)
		(visible waypoint3 waypoint12)
		(visible waypoint12 waypoint6)
		(visible waypoint6 waypoint12)
		(visible waypoint12 waypoint11)
		(visible waypoint11 waypoint12)
		(visible waypoint12 waypoint13)
		(visible waypoint13 waypoint12)
		(visible waypoint12 waypoint19)
		(visible waypoint19 waypoint12)
		(visible waypoint12 waypoint22)
		(visible waypoint22 waypoint12)
		(visible waypoint12 waypoint23)
		(visible waypoint23 waypoint12)
		(visible waypoint12 waypoint24)
		(visible waypoint24 waypoint12)
		(visible waypoint13 waypoint14)
		(visible waypoint14 waypoint13)
		(visible waypoint13 waypoint15)
		(visible waypoint15 waypoint13)
		(visible waypoint13 waypoint20)
		(visible waypoint20 waypoint13)
		(visible waypoint14 waypoint0)
		(visible waypoint0 waypoint14)
		(visible waypoint14 waypoint1)
		(visible waypoint1 waypoint14)
		(visible waypoint14 waypoint5)
		(visible waypoint5 waypoint14)
		(visible waypoint14 waypoint6)
		(visible waypoint6 waypoint14)
		(visible waypoint14 waypoint8)
		(visible waypoint8 waypoint14)
		(visible waypoint14 waypoint12)
		(visible waypoint12 waypoint14)
		(visible waypoint14 waypoint20)
		(visible waypoint20 waypoint14)
		(visible waypoint15 waypoint1)
		(visible waypoint1 waypoint15)
		(visible waypoint15 waypoint6)
		(visible waypoint6 waypoint15)
		(visible waypoint15 waypoint12)
		(visible waypoint12 waypoint15)
		(visible waypoint15 waypoint16)
		(visible waypoint16 waypoint15)
		(visible waypoint16 waypoint10)
		(visible waypoint10 waypoint16)
		(visible waypoint16 waypoint20)
		(visible waypoint20 waypoint16)
		(visible waypoint17 waypoint11)
		(visible waypoint11 waypoint17)
		(visible waypoint17 waypoint14)
		(visible waypoint14 waypoint17)
		(visible waypoint17 waypoint15)
		(visible waypoint15 waypoint17)
		(visible waypoint17 waypoint18)
		(visible waypoint18 waypoint17)
		(visible waypoint17 waypoint20)
		(visible waypoint20 waypoint17)
		(visible waypoint18 waypoint0)
		(visible waypoint0 waypoint18)
		(visible waypoint18 waypoint1)
		(visible waypoint1 waypoint18)
		(visible waypoint18 waypoint6)
		(visible waypoint6 waypoint18)
		(visible waypoint18 waypoint8)
		(visible waypoint8 waypoint18)
		(visible waypoint18 waypoint9)
		(visible waypoint9 waypoint18)
		(visible waypoint18 waypoint10)
		(visible waypoint10 waypoint18)
		(visible waypoint19 waypoint6)
		(visible waypoint6 waypoint19)
		(visible waypoint19 waypoint21)
		(visible waypoint21 waypoint19)
		(visible waypoint20 waypoint19)
		(visible waypoint19 waypoint20)
		(visible waypoint21 waypoint4)
		(visible waypoint4 waypoint21)
		(visible waypoint21 waypoint5)
		(visible waypoint5 waypoint21)
		(visible waypoint21 waypoint8)
		(visible waypoint8 waypoint21)
		(visible waypoint21 waypoint11)
		(visible waypoint11 waypoint21)
		(visible waypoint21 waypoint13)
		(visible waypoint13 waypoint21)
		(visible waypoint21 waypoint17)
		(visible waypoint17 waypoint21)
		(visible waypoint21 waypoint18)
		(visible waypoint18 waypoint21)
		(visible waypoint22 waypoint11)
		(visible waypoint11 waypoint22)
		(visible waypoint22 waypoint16)
		(visible waypoint16 waypoint22)
		(visible waypoint22 waypoint21)
		(visible waypoint21 waypoint22)
		(visible waypoint23 waypoint10)
		(visible waypoint10 waypoint23)
		(visible waypoint23 waypoint18)
		(visible waypoint18 waypoint23)
		(visible waypoint23 waypoint20)
		(visible waypoint20 waypoint23)
		(visible waypoint24 waypoint5)
		(visible waypoint5 waypoint24)
		(visible waypoint24 waypoint7)
		(visible waypoint7 waypoint24)
		(visible waypoint24 waypoint18)
		(visible waypoint18 waypoint24)
		(visible waypoint24 waypoint20)
		(visible waypoint20 waypoint24)
		(visible waypoint24 waypoint23)
		(visible waypoint23 waypoint24)
		(at_lander general waypoint1)
		(channel_free general)
		(available rover0)
		(store_of rover0store rover0)
		(equipped_for_soil_analysis rover0)
		(equipped_for_imaging rover0)
		(can_traverse rover0 waypoint22 waypoint11)
		(can_traverse rover0 waypoint11 waypoint22)
		(can_traverse rover0 waypoint22 waypoint12)
		(can_traverse rover0 waypoint12 waypoint22)
		(can_traverse rover0 waypoint22 waypoint21)
		(can_traverse rover0 waypoint21 waypoint22)
		(can_traverse rover0 waypoint11 waypoint6)
		(can_traverse rover0 waypoint6 waypoint11)
		(can_traverse rover0 waypoint11 waypoint8)
		(can_traverse rover0 waypoint8 waypoint11)
		(can_traverse rover0 waypoint11 waypoint10)
		(can_traverse rover0 waypoint10 waypoint11)
		(can_traverse rover0 waypoint11 waypoint17)
		(can_traverse rover0 waypoint17 waypoint11)
		(can_traverse rover0 waypoint11 waypoint19)
		(can_traverse rover0 waypoint19 waypoint11)
		(can_traverse rover0 waypoint12 waypoint13)
		(can_traverse rover0 waypoint13 waypoint12)
		(can_traverse rover0 waypoint12 waypoint15)
		(can_traverse rover0 waypoint15 waypoint12)
		(can_traverse rover0 waypoint12 waypoint24)
		(can_traverse rover0 waypoint24 waypoint12)
		(can_traverse rover0 waypoint21 waypoint4)
		(can_traverse rover0 waypoint4 waypoint21)
		(can_traverse rover0 waypoint21 waypoint5)
		(can_traverse rover0 waypoint5 waypoint21)
		(can_traverse rover0 waypoint21 waypoint18)
		(can_traverse rover0 waypoint18 waypoint21)
		(can_traverse rover0 waypoint6 waypoint2)
		(can_traverse rover0 waypoint2 waypoint6)
		(can_traverse rover0 waypoint6 waypoint9)
		(can_traverse rover0 waypoint9 waypoint6)
		(can_traverse rover0 waypoint6 waypoint14)
		(can_traverse rover0 waypoint14 waypoint6)
		(can_traverse rover0 waypoint8 waypoint7)
		(can_traverse rover0 waypoint7 waypoint8)
		(can_traverse rover0 waypoint17 waypoint20)
		(can_traverse rover0 waypoint20 waypoint17)
		(can_traverse rover0 waypoint13 waypoint3)
		(can_traverse rover0 waypoint3 waypoint13)
		(can_traverse rover0 waypoint15 waypoint1)
		(can_traverse rover0 waypoint1 waypoint15)
		(can_traverse rover0 waypoint15 waypoint16)
		(can_traverse rover0 waypoint16 waypoint15)
		(can_traverse rover0 waypoint24 waypoint23)
		(can_traverse rover0 waypoint23 waypoint24)
		(can_traverse rover0 waypoint5 waypoint0)
		(can_traverse rover0 waypoint0 waypoint5)
		(available rover1)
		(store_of rover1store rover1)
		(empty rover1store)
		(equipped_for_imaging rover1)
		(can_traverse rover1 waypoint4 waypoint1)
		(can_traverse rover1 waypoint1 waypoint4)
		(can_traverse rover1 waypoint4 waypoint5)
		(can_traverse rover1 waypoint5 waypoint4)
		(can_traverse rover1 waypoint4 waypoint6)
		(can_traverse rover1 waypoint6 waypoint4)
		(can_traverse rover1 waypoint4 waypoint13)
		(can_traverse rover1 waypoint13 waypoint4)
		(can_traverse rover1 waypoint4 waypoint16)
		(can_traverse rover1 waypoint16 waypoint4)
		(can_traverse rover1 waypoint4 waypoint19)
		(can_traverse rover1 waypoint19 waypoint4)
		(can_traverse rover1 waypoint4 waypoint20)
		(can_traverse rover1 waypoint20 waypoint4)
		(can_traverse rover1 waypoint4 waypoint21)
		(can_traverse rover1 waypoint21 waypoint4)
		(can_traverse rover1 waypoint1 waypoint14)
		(can_traverse rover1 waypoint14 waypoint1)
		(can_traverse rover1 waypoint1 waypoint15)
		(can_traverse rover1 waypoint15 waypoint1)
		(can_traverse rover1 waypoint1 waypoint18)
		(can_traverse rover1 waypoint18 waypoint1)
		(can_traverse rover1 waypoint5 waypoint0)
		(can_traverse rover1 waypoint0 waypoint5)
		(can_traverse rover1 waypoint5 waypoint2)
		(can_traverse rover1 waypoint2 waypoint5)
		(can_traverse rover1 waypoint5 waypoint8)
		(can_traverse rover1 waypoint8 waypoint5)
		(can_traverse rover1 waypoint5 waypoint9)
		(can_traverse rover1 waypoint9 waypoint5)
		(can_traverse rover1 waypoint5 waypoint24)
		(can_traverse rover1 waypoint24 waypoint5)
		(can_traverse rover1 waypoint6 waypoint11)
		(can_traverse rover1 waypoint11 waypoint6)
		(can_traverse rover1 waypoint6 waypoint12)
		(can_traverse rover1 waypoint12 waypoint6)
		(can_traverse rover1 waypoint13 waypoint3)
		(can_traverse rover1 waypoint3 waypoint13)
		(can_traverse rover1 waypoint16 waypoint22)
		(can_traverse rover1 waypoint22 waypoint16)
		(can_traverse rover1 waypoint20 waypoint17)
		(can_traverse rover1 waypoint17 waypoint20)
		(can_traverse rover1 waypoint18 waypoint10)
		(can_traverse rover1 waypoint10 waypoint18)
		(can_traverse rover1 waypoint18 waypoint23)
		(can_traverse rover1 waypoint23 waypoint18)
		(can_traverse rover1 waypoint0 waypoint7)
		(can_traverse rover1 waypoint7 waypoint0)
		(available rover2)
		(store_of rover2store rover2)
		(equipped_for_rock_analysis rover2)
		(equipped_for_imaging rover2)
		(can_traverse rover2 waypoint3 waypoint1)
		(can_traverse rover2 waypoint1 waypoint3)
		(can_traverse rover2 waypoint3 waypoint13)
		(can_traverse rover2 waypoint13 waypoint3)
		(can_traverse rover2 waypoint3 waypoint18)
		(can_traverse rover2 waypoint18 waypoint3)
		(can_traverse rover2 waypoint3 waypoint20)
		(can_traverse rover2 waypoint20 waypoint3)
		(can_traverse rover2 waypoint1 waypoint2)
		(can_traverse rover2 waypoint2 waypoint1)
		(can_traverse rover2 waypoint1 waypoint5)
		(can_traverse rover2 waypoint5 waypoint1)
		(can_traverse rover2 waypoint1 waypoint14)
		(can_traverse rover2 waypoint14 waypoint1)
		(can_traverse rover2 waypoint1 waypoint15)
		(can_traverse rover2 waypoint15 waypoint1)
		(can_traverse rover2 waypoint13 waypoint4)
		(can_traverse rover2 waypoint4 waypoint13)
		(can_traverse rover2 waypoint13 waypoint21)
		(can_traverse rover2 waypoint21 waypoint13)
		(can_traverse rover2 waypoint18 waypoint6)
		(can_traverse rover2 waypoint6 waypoint18)
		(can_traverse rover2 waypoint18 waypoint9)
		(can_traverse rover2 waypoint9 waypoint18)
		(can_traverse rover2 waypoint18 waypoint17)
		(can_traverse rover2 waypoint17 waypoint18)
		(can_traverse rover2 waypoint18 waypoint23)
		(can_traverse rover2 waypoint23 waypoint18)
		(can_traverse rover2 waypoint18 waypoint24)
		(can_traverse rover2 waypoint24 waypoint18)
		(can_traverse rover2 waypoint20 waypoint16)
		(can_traverse rover2 waypoint16 waypoint20)
		(can_traverse rover2 waypoint20 waypoint19)
		(can_traverse rover2 waypoint19 waypoint20)
		(can_traverse rover2 waypoint5 waypoint0)
		(can_traverse rover2 waypoint0 waypoint5)
		(can_traverse rover2 waypoint5 waypoint8)
		(can_traverse rover2 waypoint8 waypoint5)
		(can_traverse rover2 waypoint14 waypoint12)
		(can_traverse rover2 waypoint12 waypoint14)
		(can_traverse rover2 waypoint15 waypoint10)
		(can_traverse rover2 waypoint10 waypoint15)
		(can_traverse rover2 waypoint4 waypoint7)
		(can_traverse rover2 waypoint7 waypoint4)
		(can_traverse rover2 waypoint21 waypoint11)
		(can_traverse rover2 waypoint11 waypoint21)
		(can_traverse rover2 waypoint21 waypoint22)
		(can_traverse rover2 waypoint22 waypoint21)
		(available rover3)
		(store_of rover3store rover3)
		(equipped_for_soil_analysis rover3)
		(equipped_for_rock_analysis rover3)
		(can_traverse rover3 waypoint3 waypoint1)
		(can_traverse rover3 waypoint1 waypoint3)
		(can_traverse rover3 waypoint3 waypoint12)
		(can_traverse rover3 waypoint12 waypoint3)
		(can_traverse rover3 waypoint3 waypoint13)
		(can_traverse rover3 waypoint13 waypoint3)
		(can_traverse rover3 waypoint3 waypoint18)
		(can_traverse rover3 waypoint18 waypoint3)
		(can_traverse rover3 waypoint3 waypoint20)
		(can_traverse rover3 waypoint20 waypoint3)
		(can_traverse rover3 waypoint1 waypoint2)
		(can_traverse rover3 waypoint2 waypoint1)
		(can_traverse rover3 waypoint1 waypoint4)
		(can_traverse rover3 waypoint4 waypoint1)
		(can_traverse rover3 waypoint1 waypoint5)
		(can_traverse rover3 waypoint5 waypoint1)
		(can_traverse rover3 waypoint1 waypoint14)
		(can_traverse rover3 waypoint14 waypoint1)
		(can_traverse rover3 waypoint1 waypoint15)
		(can_traverse rover3 waypoint15 waypoint1)
		(can_traverse rover3 waypoint1 waypoint23)
		(can_traverse rover3 waypoint23 waypoint1)
		(can_traverse rover3 waypoint12 waypoint6)
		(can_traverse rover3 waypoint6 waypoint12)
		(can_traverse rover3 waypoint12 waypoint19)
		(can_traverse rover3 waypoint19 waypoint12)
		(can_traverse rover3 waypoint12 waypoint22)
		(can_traverse rover3 waypoint22 waypoint12)
		(can_traverse rover3 waypoint12 waypoint24)
		(can_traverse rover3 waypoint24 waypoint12)
		(can_traverse rover3 waypoint13 waypoint21)
		(can_traverse rover3 waypoint21 waypoint13)
		(can_traverse rover3 waypoint18 waypoint0)
		(can_traverse rover3 waypoint0 waypoint18)
		(can_traverse rover3 waypoint18 waypoint9)
		(can_traverse rover3 waypoint9 waypoint18)
		(can_traverse rover3 waypoint18 waypoint17)
		(can_traverse rover3 waypoint17 waypoint18)
		(can_traverse rover3 waypoint20 waypoint16)
		(can_traverse rover3 waypoint16 waypoint20)
		(can_traverse rover3 waypoint4 waypoint7)
		(can_traverse rover3 waypoint7 waypoint4)
		(can_traverse rover3 waypoint4 waypoint8)
		(can_traverse rover3 waypoint8 waypoint4)
		(can_traverse rover3 waypoint6 waypoint11)
		(can_traverse rover3 waypoint11 waypoint6)
		(can_traverse rover3 waypoint16 waypoint10)
		(can_traverse rover3 waypoint10 waypoint16)
		(available rover4)
		(store_of rover4store rover4)
		(equipped_for_rock_analysis rover4)
		(equipped_for_imaging rover4)
		(can_traverse rover4 waypoint16 waypoint2)
		(can_traverse rover4 waypoint2 waypoint16)
		(can_traverse rover4 waypoint16 waypoint4)
		(can_traverse rover4 waypoint4 waypoint16)
		(can_traverse rover4 waypoint16 waypoint5)
		(can_traverse rover4 waypoint5 waypoint16)
		(can_traverse rover4 waypoint16 waypoint7)
		(can_traverse rover4 waypoint7 waypoint16)
		(can_traverse rover4 waypoint16 waypoint8)
		(can_traverse rover4 waypoint8 waypoint16)
		(can_traverse rover4 waypoint16 waypoint10)
		(can_traverse rover4 waypoint10 waypoint16)
		(can_traverse rover4 waypoint16 waypoint15)
		(can_traverse rover4 waypoint15 waypoint16)
		(can_traverse rover4 waypoint16 waypoint22)
		(can_traverse rover4 waypoint22 waypoint16)
		(can_traverse rover4 waypoint2 waypoint9)
		(can_traverse rover4 waypoint9 waypoint2)
		(can_traverse rover4 waypoint2 waypoint14)
		(can_traverse rover4 waypoint14 waypoint2)
		(can_traverse rover4 waypoint4 waypoint1)
		(can_traverse rover4 waypoint1 waypoint4)
		(can_traverse rover4 waypoint4 waypoint13)
		(can_traverse rover4 waypoint13 waypoint4)
		(can_traverse rover4 waypoint4 waypoint20)
		(can_traverse rover4 waypoint20 waypoint4)
		(can_traverse rover4 waypoint5 waypoint0)
		(can_traverse rover4 waypoint0 waypoint5)
		(can_traverse rover4 waypoint5 waypoint21)
		(can_traverse rover4 waypoint21 waypoint5)
		(can_traverse rover4 waypoint5 waypoint24)
		(can_traverse rover4 waypoint24 waypoint5)
		(can_traverse rover4 waypoint7 waypoint17)
		(can_traverse rover4 waypoint17 waypoint7)
		(can_traverse rover4 waypoint8 waypoint11)
		(can_traverse rover4 waypoint11 waypoint8)
		(can_traverse rover4 waypoint8 waypoint18)
		(can_traverse rover4 waypoint18 waypoint8)
		(can_traverse rover4 waypoint15 waypoint3)
		(can_traverse rover4 waypoint3 waypoint15)
		(can_traverse rover4 waypoint15 waypoint6)
		(can_traverse rover4 waypoint6 waypoint15)
		(can_traverse rover4 waypoint15 waypoint12)
		(can_traverse rover4 waypoint12 waypoint15)
		(can_traverse rover4 waypoint20 waypoint23)
		(can_traverse rover4 waypoint23 waypoint20)
		(can_traverse rover4 waypoint21 waypoint19)
		(can_traverse rover4 waypoint19 waypoint21)
		(available rover5)
		(store_of rover5store rover5)
		(empty rover5store)
		(equipped_for_imaging rover5)
		(can_traverse rover5 waypoint10 waypoint11)
		(can_traverse rover5 waypoint11 waypoint10)
		(can_traverse rover5 waypoint10 waypoint15)
		(can_traverse rover5 waypoint15 waypoint10)
		(can_traverse rover5 waypoint10 waypoint16)
		(can_traverse rover5 waypoint16 waypoint10)
		(can_traverse rover5 waypoint10 waypoint18)
		(can_traverse rover5 waypoint18 waypoint10)
		(can_traverse rover5 waypoint10 waypoint23)
		(can_traverse rover5 waypoint23 waypoint10)
		(can_traverse rover5 waypoint11 waypoint6)
		(can_traverse rover5 waypoint6 waypoint11)
		(can_traverse rover5 waypoint11 waypoint8)
		(can_traverse rover5 waypoint8 waypoint11)
		(can_traverse rover5 waypoint11 waypoint12)
		(can_traverse rover5 waypoint12 waypoint11)
		(can_traverse rover5 waypoint11 waypoint21)
		(can_traverse rover5 waypoint21 waypoint11)
		(can_traverse rover5 waypoint11 waypoint22)
		(can_traverse rover5 waypoint22 waypoint11)
		(can_traverse rover5 waypoint15 waypoint0)
		(can_traverse rover5 waypoint0 waypoint15)
		(can_traverse rover5 waypoint15 waypoint1)
		(can_traverse rover5 waypoint1 waypoint15)
		(can_traverse rover5 waypoint15 waypoint13)
		(can_traverse rover5 waypoint13 waypoint15)
		(can_traverse rover5 waypoint15 waypoint17)
		(can_traverse rover5 waypoint17 waypoint15)
		(can_traverse rover5 waypoint16 waypoint2)
		(can_traverse rover5 waypoint2 waypoint16)
		(can_traverse rover5 waypoint16 waypoint4)
		(can_traverse rover5 waypoint4 waypoint16)
		(can_traverse rover5 waypoint16 waypoint7)
		(can_traverse rover5 waypoint7 waypoint16)
		(can_traverse rover5 waypoint18 waypoint3)
		(can_traverse rover5 waypoint3 waypoint18)
		(can_traverse rover5 waypoint23 waypoint20)
		(can_traverse rover5 waypoint20 waypoint23)
		(can_traverse rover5 waypoint23 waypoint24)
		(can_traverse rover5 waypoint24 waypoint23)
		(can_traverse rover5 waypoint6 waypoint5)
		(can_traverse rover5 waypoint5 waypoint6)
		(can_traverse rover5 waypoint6 waypoint9)
		(can_traverse rover5 waypoint9 waypoint6)
		(can_traverse rover5 waypoint6 waypoint14)
		(can_traverse rover5 waypoint14 waypoint6)
		(can_traverse rover5 waypoint6 waypoint19)
		(can_traverse rover5 waypoint19 waypoint6)
		(available rover6)
		(store_of rover6store rover6)
		(equipped_for_soil_analysis rover6)
		(equipped_for_imaging rover6)
		(can_traverse rover6 waypoint4 waypoint1)
		(can_traverse rover6 waypoint1 waypoint4)
		(can_traverse rover6 waypoint4 waypoint6)
		(can_traverse rover6 waypoint6 waypoint4)
		(can_traverse rover6 waypoint4 waypoint8)
		(can_traverse rover6 waypoint8 waypoint4)
		(can_traverse rover6 waypoint4 waypoint13)
		(can_traverse rover6 waypoint13 waypoint4)
		(can_traverse rover6 waypoint4 waypoint16)
		(can_traverse rover6 waypoint16 waypoint4)
		(can_traverse rover6 waypoint4 waypoint19)
		(can_traverse rover6 waypoint19 waypoint4)
		(can_traverse rover6 waypoint4 waypoint20)
		(can_traverse rover6 waypoint20 waypoint4)
		(can_traverse rover6 waypoint1 waypoint3)
		(can_traverse rover6 waypoint3 waypoint1)
		(can_traverse rover6 waypoint1 waypoint5)
		(can_traverse rover6 waypoint5 waypoint1)
		(can_traverse rover6 waypoint1 waypoint14)
		(can_traverse rover6 waypoint14 waypoint1)
		(can_traverse rover6 waypoint1 waypoint18)
		(can_traverse rover6 waypoint18 waypoint1)
		(can_traverse rover6 waypoint1 waypoint23)
		(can_traverse rover6 waypoint23 waypoint1)
		(can_traverse rover6 waypoint6 waypoint2)
		(can_traverse rover6 waypoint2 waypoint6)
		(can_traverse rover6 waypoint6 waypoint9)
		(can_traverse rover6 waypoint9 waypoint6)
		(can_traverse rover6 waypoint6 waypoint11)
		(can_traverse rover6 waypoint11 waypoint6)
		(can_traverse rover6 waypoint6 waypoint12)
		(can_traverse rover6 waypoint12 waypoint6)
		(can_traverse rover6 waypoint6 waypoint15)
		(can_traverse rover6 waypoint15 waypoint6)
		(can_traverse rover6 waypoint6 waypoint21)
		(can_traverse rover6 waypoint21 waypoint6)
		(can_traverse rover6 waypoint8 waypoint17)
		(can_traverse rover6 waypoint17 waypoint8)
		(can_traverse rover6 waypoint16 waypoint7)
		(can_traverse rover6 waypoint7 waypoint16)
		(can_traverse rover6 waypoint16 waypoint10)
		(can_traverse rover6 waypoint10 waypoint16)
		(can_traverse rover6 waypoint16 waypoint22)
		(can_traverse rover6 waypoint22 waypoint16)
		(can_traverse rover6 waypoint20 waypoint24)
		(can_traverse rover6 waypoint24 waypoint20)
		(can_traverse rover6 waypoint23 waypoint0)
		(can_traverse rover6 waypoint0 waypoint23)
		(available rover7)
		(store_of rover7store rover7)
		(equipped_for_rock_analysis rover7)
		(can_traverse rover7 waypoint16 waypoint2)
		(can_traverse rover7 waypoint2 waypoint16)
		(can_traverse rover7 waypoint16 waypoint4)
		(can_traverse rover7 waypoint4 waypoint16)
		(can_traverse rover7 waypoint16 waypoint8)
		(can_traverse rover7 waypoint8 waypoint16)
		(can_traverse rover7 waypoint16 waypoint10)
		(can_traverse rover7 waypoint10 waypoint16)
		(can_traverse rover7 waypoint16 waypoint20)
		(can_traverse rover7 waypoint20 waypoint16)
		(can_traverse rover7 waypoint2 waypoint5)
		(can_traverse rover7 waypoint5 waypoint2)
		(can_traverse rover7 waypoint2 waypoint6)
		(can_traverse rover7 waypoint6 waypoint2)
		(can_traverse rover7 waypoint2 waypoint9)
		(can_traverse rover7 waypoint9 waypoint2)
		(can_traverse rover7 waypoint2 waypoint14)
		(can_traverse rover7 waypoint14 waypoint2)
		(can_traverse rover7 waypoint2 waypoint23)
		(can_traverse rover7 waypoint23 waypoint2)
		(can_traverse rover7 waypoint4 waypoint13)
		(can_traverse rover7 waypoint13 waypoint4)
		(can_traverse rover7 waypoint4 waypoint19)
		(can_traverse rover7 waypoint19 waypoint4)
		(can_traverse rover7 waypoint8 waypoint7)
		(can_traverse rover7 waypoint7 waypoint8)
		(can_traverse rover7 waypoint8 waypoint17)
		(can_traverse rover7 waypoint17 waypoint8)
		(can_traverse rover7 waypoint8 waypoint18)
		(can_traverse rover7 waypoint18 waypoint8)
		(can_traverse rover7 waypoint10 waypoint11)
		(can_traverse rover7 waypoint11 waypoint10)
		(can_traverse rover7 waypoint10 waypoint15)
		(can_traverse rover7 waypoint15 waypoint10)
		(can_traverse rover7 waypoint20 waypoint3)
		(can_traverse rover7 waypoint3 waypoint20)
		(can_traverse rover7 waypoint20 waypoint24)
		(can_traverse rover7 waypoint24 waypoint20)
		(can_traverse rover7 waypoint5 waypoint21)
		(can_traverse rover7 waypoint21 waypoint5)
		(can_traverse rover7 waypoint6 waypoint12)
		(can_traverse rover7 waypoint12 waypoint6)
		(can_traverse rover7 waypoint14 waypoint0)
		(can_traverse rover7 waypoint0 waypoint14)
		(can_traverse rover7 waypoint14 waypoint1)
		(can_traverse rover7 waypoint1 waypoint14)
		(on_board camera0 rover0)
		(calibration_target camera0 objective6)
		(supports camera0 colour)
		(supports camera0 high_res)
		(on_board camera1 rover1)
		(calibration_target camera1 objective1)
		(supports camera1 colour)
		(supports camera1 high_res)
		(on_board camera2 rover4)
		(calibration_target camera2 objective0)
		(supports camera2 colour)
		(supports camera2 high_res)
		(supports camera2 low_res)
		(on_board camera3 rover2)
		(calibration_target camera3 objective2)
		(supports camera3 high_res)
		(supports camera3 low_res)
		(on_board camera4 rover1)
		(calibration_target camera4 objective4)
		(supports camera4 colour)
		(supports camera4 high_res)
		(on_board camera5 rover5)
		(calibration_target camera5 objective4)
		(supports camera5 high_res)
		(on_board camera6 rover6)
		(calibration_target camera6 objective5)
		(supports camera6 colour)
		(supports camera6 high_res)
		(supports camera6 low_res)
		(visible_from objective0 waypoint0)
		(visible_from objective0 waypoint1)
		(visible_from objective0 waypoint2)
		(visible_from objective0 waypoint3)
		(visible_from objective0 waypoint4)
		(visible_from objective0 waypoint5)
		(visible_from objective0 waypoint6)
		(visible_from objective0 waypoint7)
		(visible_from objective0 waypoint8)
		(visible_from objective0 waypoint9)
		(visible_from objective0 waypoint10)
		(visible_from objective0 waypoint11)
		(visible_from objective0 waypoint12)
		(visible_from objective0 waypoint13)
		(visible_from objective0 waypoint14)
		(visible_from objective0 waypoint15)
		(visible_from objective0 waypoint16)
		(visible_from objective1 waypoint0)
		(visible_from objective1 waypoint1)
		(visible_from objective1 waypoint2)
		(visible_from objective1 waypoint3)
		(visible_from objective2 waypoint0)
		(visible_from objective2 waypoint1)
		(visible_from objective2 waypoint2)
		(visible_from objective2 waypoint3)
		(visible_from objective2 waypoint4)
		(visible_from objective2 waypoint5)
		(visible_from objective2 waypoint6)
		(visible_from objective2 waypoint7)
		(visible_from objective2 waypoint8)
		(visible_from objective2 waypoint9)
		(visible_from objective2 waypoint10)
		(visible_from objective2 waypoint11)
		(visible_from objective2 waypoint12)
		(visible_from objective2 waypoint13)
		(visible_from objective2 waypoint14)
		(visible_from objective2 waypoint15)
		(visible_from objective2 waypoint16)
		(visible_from objective2 waypoint17)
		(visible_from objective3 waypoint0)
		(visible_from objective3 waypoint1)
		(visible_from objective3 waypoint2)
		(visible_from objective3 waypoint3)
		(visible_from objective3 waypoint4)
		(visible_from objective3 waypoint5)
		(visible_from objective3 waypoint6)
		(visible_from objective3 waypoint7)
		(visible_from objective3 waypoint8)
		(visible_from objective3 waypoint9)
		(visible_from objective3 waypoint10)
		(visible_from objective3 waypoint11)
		(visible_from objective3 waypoint12)
		(visible_from objective3 waypoint13)
		(visible_from objective3 waypoint14)
		(visible_from objective3 waypoint15)
		(visible_from objective3 waypoint16)
		(visible_from objective3 waypoint17)
		(visible_from objective3 waypoint18)
		(visible_from objective3 waypoint19)
		(visible_from objective4 waypoint0)
		(visible_from objective4 waypoint1)
		(visible_from objective4 waypoint2)
		(visible_from objective4 waypoint3)
		(visible_from objective4 waypoint4)
		(visible_from objective4 waypoint5)
		(visible_from objective4 waypoint6)
		(visible_from objective4 waypoint7)
		(visible_from objective4 waypoint8)
		(visible_from objective4 waypoint9)
		(visible_from objective4 waypoint10)
		(visible_from objective4 waypoint11)
		(visible_from objective4 waypoint12)
		(visible_from objective4 waypoint13)
		(visible_from objective4 waypoint14)
		(visible_from objective4 waypoint15)
		(visible_from objective4 waypoint16)
		(visible_from objective4 waypoint17)
		(visible_from objective4 waypoint18)
		(visible_from objective4 waypoint19)
		(visible_from objective4 waypoint20)
		(visible_from objective4 waypoint21)
		(visible_from objective4 waypoint22)
		(visible_from objective4 waypoint23)
		(visible_from objective5 waypoint0)
		(visible_from objective5 waypoint1)
		(visible_from objective5 waypoint2)
		(visible_from objective5 waypoint3)
		(visible_from objective5 waypoint4)
		(visible_from objective5 waypoint5)
		(visible_from objective5 waypoint6)
		(visible_from objective5 waypoint7)
		(visible_from objective5 waypoint8)
		(visible_from objective5 waypoint9)
		(visible_from objective5 waypoint10)
		(visible_from objective5 waypoint11)
		(visible_from objective5 waypoint12)
		(visible_from objective5 waypoint13)
		(visible_from objective5 waypoint14)
		(visible_from objective6 waypoint0)
		(visible_from objective6 waypoint1)
		(visible_from objective6 waypoint2)
		(visible_from objective6 waypoint3)
		(visible_from objective6 waypoint4)
		(visible_from objective6 waypoint5)
		(visible_from objective6 waypoint6)
		(visible_from objective6 waypoint7)
		(visible_from objective6 waypoint8)
		(visible_from objective6 waypoint9)
		(visible_from objective6 waypoint10)
		(visible_from objective7 waypoint0)
		(visible_from objective7 waypoint1)
		(visible_from objective7 waypoint2)
		(visible_from objective7 waypoint3)
		(visible_from objective7 waypoint4)
		(visible_from objective7 waypoint5)
		(visible_from objective7 waypoint6)
		(visible_from objective7 waypoint7)
		(visible_from objective7 waypoint8)
		(visible_from objective7 waypoint9)
		(visible_from objective7 waypoint10)
		(visible_from objective7 waypoint11)
		(visible_from objective7 waypoint12)
		(visible_from objective7 waypoint13)
		(visible_from objective7 waypoint14)
		(visible_from objective7 waypoint15)
		(visible_from objective7 waypoint16)
		(visible_from objective7 waypoint17)
		(visible_from objective7 waypoint18)
		(visible_from objective7 waypoint19)
		(visible_from objective7 waypoint20)
		)

(:goal (and
(communicated_soil_data waypoint8)
(communicated_soil_data waypoint23)
(communicated_soil_data waypoint7)
(communicated_soil_data waypoint11)
(communicated_soil_data waypoint13)
(communicated_soil_data waypoint20)
(communicated_rock_data waypoint7)
(communicated_rock_data waypoint14)
(communicated_rock_data waypoint22)
(communicated_rock_data waypoint16)
(communicated_rock_data waypoint12)
(communicated_rock_data waypoint10)
(communicated_rock_data waypoint9)
(communicated_rock_data waypoint21)
(communicated_rock_data waypoint18)
(communicated_image_data objective2 high_res)
(communicated_image_data objective0 high_res)
(communicated_image_data objective3 colour)
(communicated_image_data objective7 colour)
(communicated_image_data objective5 high_res)
	)
)
)

