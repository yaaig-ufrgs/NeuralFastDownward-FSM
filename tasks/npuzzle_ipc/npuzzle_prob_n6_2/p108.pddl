(define (problem n-puzzle-6)
  (:domain n-puzzle-typed)
  (:objects p_1_1 p_1_2 p_1_3 p_1_4 p_1_5 p_1_6 p_2_1 p_2_2 p_2_3 p_2_4 p_2_5 p_2_6 p_3_1 p_3_2 p_3_3 p_3_4 p_3_5 p_3_6 p_4_1 p_4_2 p_4_3 p_4_4 p_4_5 p_4_6 p_5_1 p_5_2 p_5_3 p_5_4 p_5_5 p_5_6 p_6_1 p_6_2 p_6_3 p_6_4 p_6_5 p_6_6 - position t_1 t_2 t_3 t_4 t_5 t_6 t_7 t_8 t_9 t_10 t_11 t_12 t_13 t_14 t_15 t_16 t_17 t_18 t_19 t_20 t_21 t_22 t_23 t_24 t_25 t_26 t_27 t_28 t_29 t_30 t_31 t_32 t_33 t_34 t_35 - tile)
  		(:init
		(at t_1 p_1_4)
		(at t_10 p_3_4)
		(at t_11 p_2_1)
		(at t_12 p_4_4)
		(at t_13 p_4_6)
		(at t_14 p_3_2)
		(at t_15 p_5_4)
		(at t_16 p_5_1)
		(at t_17 p_4_2)
		(at t_18 p_1_6)
		(at t_19 p_3_5)
		(at t_2 p_1_1)
		(at t_20 p_3_1)
		(at t_21 p_5_3)
		(at t_22 p_4_3)
		(at t_23 p_1_5)
		(at t_24 p_1_2)
		(at t_25 p_5_2)
		(at t_26 p_3_6)
		(at t_27 p_5_5)
		(at t_28 p_6_6)
		(at t_29 p_4_1)
		(at t_3 p_3_3)
		(at t_30 p_6_1)
		(at t_31 p_2_4)
		(at t_32 p_2_6)
		(at t_33 p_2_2)
		(at t_34 p_6_5)
		(at t_35 p_5_6)
		(at t_4 p_1_3)
		(at t_5 p_6_3)
		(at t_6 p_6_4)
		(at t_7 p_2_3)
		(at t_8 p_2_5)
		(at t_9 p_4_5)
		(empty p_6_2)
		(neighbor p_1_1 p_1_2)
		(neighbor p_1_2 p_1_1)
		(neighbor p_1_2 p_1_3)
		(neighbor p_1_3 p_1_2)
		(neighbor p_1_3 p_1_4)
		(neighbor p_1_4 p_1_3)
		(neighbor p_1_4 p_1_5)
		(neighbor p_1_5 p_1_4)
		(neighbor p_1_5 p_1_6)
		(neighbor p_1_6 p_1_5)
		(neighbor p_2_1 p_2_2)
		(neighbor p_2_2 p_2_1)
		(neighbor p_2_2 p_2_3)
		(neighbor p_2_3 p_2_2)
		(neighbor p_2_3 p_2_4)
		(neighbor p_2_4 p_2_3)
		(neighbor p_2_4 p_2_5)
		(neighbor p_2_5 p_2_4)
		(neighbor p_2_5 p_2_6)
		(neighbor p_2_6 p_2_5)
		(neighbor p_3_1 p_3_2)
		(neighbor p_3_2 p_3_1)
		(neighbor p_3_2 p_3_3)
		(neighbor p_3_3 p_3_2)
		(neighbor p_3_3 p_3_4)
		(neighbor p_3_4 p_3_3)
		(neighbor p_3_4 p_3_5)
		(neighbor p_3_5 p_3_4)
		(neighbor p_3_5 p_3_6)
		(neighbor p_3_6 p_3_5)
		(neighbor p_4_1 p_4_2)
		(neighbor p_4_2 p_4_1)
		(neighbor p_4_2 p_4_3)
		(neighbor p_4_3 p_4_2)
		(neighbor p_4_3 p_4_4)
		(neighbor p_4_4 p_4_3)
		(neighbor p_4_4 p_4_5)
		(neighbor p_4_5 p_4_4)
		(neighbor p_4_5 p_4_6)
		(neighbor p_4_6 p_4_5)
		(neighbor p_5_1 p_5_2)
		(neighbor p_5_2 p_5_1)
		(neighbor p_5_2 p_5_3)
		(neighbor p_5_3 p_5_2)
		(neighbor p_5_3 p_5_4)
		(neighbor p_5_4 p_5_3)
		(neighbor p_5_4 p_5_5)
		(neighbor p_5_5 p_5_4)
		(neighbor p_5_5 p_5_6)
		(neighbor p_5_6 p_5_5)
		(neighbor p_6_1 p_6_2)
		(neighbor p_6_2 p_6_1)
		(neighbor p_6_2 p_6_3)
		(neighbor p_6_3 p_6_2)
		(neighbor p_6_3 p_6_4)
		(neighbor p_6_4 p_6_3)
		(neighbor p_6_4 p_6_5)
		(neighbor p_6_5 p_6_4)
		(neighbor p_6_5 p_6_6)
		(neighbor p_6_6 p_6_5)
		(neighbor p_1_1 p_2_1)
		(neighbor p_2_1 p_1_1)
		(neighbor p_1_2 p_2_2)
		(neighbor p_2_2 p_1_2)
		(neighbor p_1_3 p_2_3)
		(neighbor p_2_3 p_1_3)
		(neighbor p_1_4 p_2_4)
		(neighbor p_2_4 p_1_4)
		(neighbor p_1_5 p_2_5)
		(neighbor p_2_5 p_1_5)
		(neighbor p_1_6 p_2_6)
		(neighbor p_2_6 p_1_6)
		(neighbor p_2_1 p_3_1)
		(neighbor p_3_1 p_2_1)
		(neighbor p_2_2 p_3_2)
		(neighbor p_3_2 p_2_2)
		(neighbor p_2_3 p_3_3)
		(neighbor p_3_3 p_2_3)
		(neighbor p_2_4 p_3_4)
		(neighbor p_3_4 p_2_4)
		(neighbor p_2_5 p_3_5)
		(neighbor p_3_5 p_2_5)
		(neighbor p_2_6 p_3_6)
		(neighbor p_3_6 p_2_6)
		(neighbor p_3_1 p_4_1)
		(neighbor p_4_1 p_3_1)
		(neighbor p_3_2 p_4_2)
		(neighbor p_4_2 p_3_2)
		(neighbor p_3_3 p_4_3)
		(neighbor p_4_3 p_3_3)
		(neighbor p_3_4 p_4_4)
		(neighbor p_4_4 p_3_4)
		(neighbor p_3_5 p_4_5)
		(neighbor p_4_5 p_3_5)
		(neighbor p_3_6 p_4_6)
		(neighbor p_4_6 p_3_6)
		(neighbor p_4_1 p_5_1)
		(neighbor p_5_1 p_4_1)
		(neighbor p_4_2 p_5_2)
		(neighbor p_5_2 p_4_2)
		(neighbor p_4_3 p_5_3)
		(neighbor p_5_3 p_4_3)
		(neighbor p_4_4 p_5_4)
		(neighbor p_5_4 p_4_4)
		(neighbor p_4_5 p_5_5)
		(neighbor p_5_5 p_4_5)
		(neighbor p_4_6 p_5_6)
		(neighbor p_5_6 p_4_6)
		(neighbor p_5_1 p_6_1)
		(neighbor p_6_1 p_5_1)
		(neighbor p_5_2 p_6_2)
		(neighbor p_6_2 p_5_2)
		(neighbor p_5_3 p_6_3)
		(neighbor p_6_3 p_5_3)
		(neighbor p_5_4 p_6_4)
		(neighbor p_6_4 p_5_4)
		(neighbor p_5_5 p_6_5)
		(neighbor p_6_5 p_5_5)
		(neighbor p_5_6 p_6_6)
		(neighbor p_6_6 p_5_6))
  (:goal (and
    (at t_1 p_1_1)
    (at t_2 p_1_2)
    (at t_3 p_1_3)
    (at t_4 p_1_4)
    (at t_5 p_1_5)
    (at t_6 p_1_6)
    (at t_7 p_2_1)
    (at t_8 p_2_2)
    (at t_9 p_2_3)
    (at t_10 p_2_4)
    (at t_11 p_2_5)
    (at t_12 p_2_6)
    (at t_13 p_3_1)
    (at t_14 p_3_2)
    (at t_15 p_3_3)
    (at t_16 p_3_4)
    (at t_17 p_3_5)
    (at t_18 p_3_6)
    (at t_19 p_4_1)
    (at t_20 p_4_2)
    (at t_21 p_4_3)
    (at t_22 p_4_4)
    (at t_23 p_4_5)
    (at t_24 p_4_6)
    (at t_25 p_5_1)
    (at t_26 p_5_2)
    (at t_27 p_5_3)
    (at t_28 p_5_4)
    (at t_29 p_5_5)
    (at t_30 p_5_6)
    (at t_31 p_6_1)
    (at t_32 p_6_2)
    (at t_33 p_6_3)
    (at t_34 p_6_4)
    (at t_35 p_6_5))))

