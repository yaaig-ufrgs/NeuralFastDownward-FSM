(define (problem strips-grid-y-4)
   (:domain grid)
   (:objects node0-0 node0-1 node0-2 node0-3 node0-4 node0-5
             node0-6 node0-7 node1-0 node1-1 node1-2 node1-3 node1-4
             node1-5 node1-6 node1-7 node2-0 node2-1 node2-2 node2-3
             node2-4 node2-5 node2-6 node2-7 node3-0 node3-1 node3-2
             node3-3 node3-4 node3-5 node3-6 node3-7 node4-0 node4-1
             node4-2 node4-3 node4-4 node4-5 node4-6 node4-7 node5-0
             node5-1 node5-2 node5-3 node5-4 node5-5 node5-6 node5-7
             node6-0 node6-1 node6-2 node6-3 node6-4 node6-5 node6-6
             node6-7 node7-0 node7-1 node7-2 node7-3 node7-4 node7-5
             node7-6 node7-7 triangle diamond square circle key0 key1 key2
             key3 key4 key5 key6 key7 key8 key9 key10 key11)
   		(:init
		(arm-empty )
		(at key0 node1-6)
		(at key1 node1-2)
		(at key10 node3-2)
		(at key11 node5-0)
		(at key2 node2-2)
		(at key3 node0-3)
		(at key4 node5-1)
		(at key5 node5-2)
		(at key6 node7-2)
		(at key7 node1-4)
		(at key8 node0-7)
		(at key9 node3-2)
		(at-robot node0-4)
		(locked node5-1)
		(locked node5-2)
		(locked node5-3)
		(locked node6-1)
		(locked node6-3)
		(locked node7-1)
		(locked node7-2)
		(locked node7-3)
		(arm-empty)
		(place node0-0)
		(place node0-1)
		(place node0-2)
		(place node0-3)
		(place node0-4)
		(place node0-5)
		(place node0-6)
		(place node0-7)
		(place node1-0)
		(place node1-1)
		(place node1-2)
		(place node1-3)
		(place node1-4)
		(place node1-5)
		(place node1-6)
		(place node1-7)
		(place node2-0)
		(place node2-1)
		(place node2-2)
		(place node2-3)
		(place node2-4)
		(place node2-5)
		(place node2-6)
		(place node2-7)
		(place node3-0)
		(place node3-1)
		(place node3-2)
		(place node3-3)
		(place node3-4)
		(place node3-5)
		(place node3-6)
		(place node3-7)
		(place node4-0)
		(place node4-1)
		(place node4-2)
		(place node4-3)
		(place node4-4)
		(place node4-5)
		(place node4-6)
		(place node4-7)
		(place node5-0)
		(place node5-1)
		(place node5-2)
		(place node5-3)
		(place node5-4)
		(place node5-5)
		(place node5-6)
		(place node5-7)
		(place node6-0)
		(place node6-1)
		(place node6-2)
		(place node6-3)
		(place node6-4)
		(place node6-5)
		(place node6-6)
		(place node6-7)
		(place node7-0)
		(place node7-1)
		(place node7-2)
		(place node7-3)
		(place node7-4)
		(place node7-5)
		(place node7-6)
		(place node7-7)
		(shape triangle)
		(shape diamond)
		(shape square)
		(shape circle)
		(conn node0-0 node1-0)
		(conn node0-0 node0-1)
		(conn node0-1 node1-1)
		(conn node0-1 node0-2)
		(conn node0-1 node0-0)
		(conn node0-2 node1-2)
		(conn node0-2 node0-3)
		(conn node0-2 node0-1)
		(conn node0-3 node1-3)
		(conn node0-3 node0-4)
		(conn node0-3 node0-2)
		(conn node0-4 node1-4)
		(conn node0-4 node0-5)
		(conn node0-4 node0-3)
		(conn node0-5 node1-5)
		(conn node0-5 node0-6)
		(conn node0-5 node0-4)
		(conn node0-6 node1-6)
		(conn node0-6 node0-7)
		(conn node0-6 node0-5)
		(conn node0-7 node1-7)
		(conn node0-7 node0-6)
		(conn node1-0 node2-0)
		(conn node1-0 node0-0)
		(conn node1-0 node1-1)
		(conn node1-1 node2-1)
		(conn node1-1 node0-1)
		(conn node1-1 node1-2)
		(conn node1-1 node1-0)
		(conn node1-2 node2-2)
		(conn node1-2 node0-2)
		(conn node1-2 node1-3)
		(conn node1-2 node1-1)
		(conn node1-3 node2-3)
		(conn node1-3 node0-3)
		(conn node1-3 node1-4)
		(conn node1-3 node1-2)
		(conn node1-4 node2-4)
		(conn node1-4 node0-4)
		(conn node1-4 node1-5)
		(conn node1-4 node1-3)
		(conn node1-5 node2-5)
		(conn node1-5 node0-5)
		(conn node1-5 node1-6)
		(conn node1-5 node1-4)
		(conn node1-6 node2-6)
		(conn node1-6 node0-6)
		(conn node1-6 node1-7)
		(conn node1-6 node1-5)
		(conn node1-7 node2-7)
		(conn node1-7 node0-7)
		(conn node1-7 node1-6)
		(conn node2-0 node3-0)
		(conn node2-0 node1-0)
		(conn node2-0 node2-1)
		(conn node2-1 node3-1)
		(conn node2-1 node1-1)
		(conn node2-1 node2-2)
		(conn node2-1 node2-0)
		(conn node2-2 node3-2)
		(conn node2-2 node1-2)
		(conn node2-2 node2-3)
		(conn node2-2 node2-1)
		(conn node2-3 node3-3)
		(conn node2-3 node1-3)
		(conn node2-3 node2-4)
		(conn node2-3 node2-2)
		(conn node2-4 node3-4)
		(conn node2-4 node1-4)
		(conn node2-4 node2-5)
		(conn node2-4 node2-3)
		(conn node2-5 node3-5)
		(conn node2-5 node1-5)
		(conn node2-5 node2-6)
		(conn node2-5 node2-4)
		(conn node2-6 node3-6)
		(conn node2-6 node1-6)
		(conn node2-6 node2-7)
		(conn node2-6 node2-5)
		(conn node2-7 node3-7)
		(conn node2-7 node1-7)
		(conn node2-7 node2-6)
		(conn node3-0 node4-0)
		(conn node3-0 node2-0)
		(conn node3-0 node3-1)
		(conn node3-1 node4-1)
		(conn node3-1 node2-1)
		(conn node3-1 node3-2)
		(conn node3-1 node3-0)
		(conn node3-2 node4-2)
		(conn node3-2 node2-2)
		(conn node3-2 node3-3)
		(conn node3-2 node3-1)
		(conn node3-3 node4-3)
		(conn node3-3 node2-3)
		(conn node3-3 node3-4)
		(conn node3-3 node3-2)
		(conn node3-4 node4-4)
		(conn node3-4 node2-4)
		(conn node3-4 node3-5)
		(conn node3-4 node3-3)
		(conn node3-5 node4-5)
		(conn node3-5 node2-5)
		(conn node3-5 node3-6)
		(conn node3-5 node3-4)
		(conn node3-6 node4-6)
		(conn node3-6 node2-6)
		(conn node3-6 node3-7)
		(conn node3-6 node3-5)
		(conn node3-7 node4-7)
		(conn node3-7 node2-7)
		(conn node3-7 node3-6)
		(conn node4-0 node5-0)
		(conn node4-0 node3-0)
		(conn node4-0 node4-1)
		(conn node4-1 node5-1)
		(conn node4-1 node3-1)
		(conn node4-1 node4-2)
		(conn node4-1 node4-0)
		(conn node4-2 node5-2)
		(conn node4-2 node3-2)
		(conn node4-2 node4-3)
		(conn node4-2 node4-1)
		(conn node4-3 node5-3)
		(conn node4-3 node3-3)
		(conn node4-3 node4-4)
		(conn node4-3 node4-2)
		(conn node4-4 node5-4)
		(conn node4-4 node3-4)
		(conn node4-4 node4-5)
		(conn node4-4 node4-3)
		(conn node4-5 node5-5)
		(conn node4-5 node3-5)
		(conn node4-5 node4-6)
		(conn node4-5 node4-4)
		(conn node4-6 node5-6)
		(conn node4-6 node3-6)
		(conn node4-6 node4-7)
		(conn node4-6 node4-5)
		(conn node4-7 node5-7)
		(conn node4-7 node3-7)
		(conn node4-7 node4-6)
		(conn node5-0 node6-0)
		(conn node5-0 node4-0)
		(conn node5-0 node5-1)
		(conn node5-1 node6-1)
		(conn node5-1 node4-1)
		(conn node5-1 node5-2)
		(conn node5-1 node5-0)
		(conn node5-2 node6-2)
		(conn node5-2 node4-2)
		(conn node5-2 node5-3)
		(conn node5-2 node5-1)
		(conn node5-3 node6-3)
		(conn node5-3 node4-3)
		(conn node5-3 node5-4)
		(conn node5-3 node5-2)
		(conn node5-4 node6-4)
		(conn node5-4 node4-4)
		(conn node5-4 node5-5)
		(conn node5-4 node5-3)
		(conn node5-5 node6-5)
		(conn node5-5 node4-5)
		(conn node5-5 node5-6)
		(conn node5-5 node5-4)
		(conn node5-6 node6-6)
		(conn node5-6 node4-6)
		(conn node5-6 node5-7)
		(conn node5-6 node5-5)
		(conn node5-7 node6-7)
		(conn node5-7 node4-7)
		(conn node5-7 node5-6)
		(conn node6-0 node7-0)
		(conn node6-0 node5-0)
		(conn node6-0 node6-1)
		(conn node6-1 node7-1)
		(conn node6-1 node5-1)
		(conn node6-1 node6-2)
		(conn node6-1 node6-0)
		(conn node6-2 node7-2)
		(conn node6-2 node5-2)
		(conn node6-2 node6-3)
		(conn node6-2 node6-1)
		(conn node6-3 node7-3)
		(conn node6-3 node5-3)
		(conn node6-3 node6-4)
		(conn node6-3 node6-2)
		(conn node6-4 node7-4)
		(conn node6-4 node5-4)
		(conn node6-4 node6-5)
		(conn node6-4 node6-3)
		(conn node6-5 node7-5)
		(conn node6-5 node5-5)
		(conn node6-5 node6-6)
		(conn node6-5 node6-4)
		(conn node6-6 node7-6)
		(conn node6-6 node5-6)
		(conn node6-6 node6-7)
		(conn node6-6 node6-5)
		(conn node6-7 node7-7)
		(conn node6-7 node5-7)
		(conn node6-7 node6-6)
		(conn node7-0 node6-0)
		(conn node7-0 node7-1)
		(conn node7-1 node6-1)
		(conn node7-1 node7-2)
		(conn node7-1 node7-0)
		(conn node7-2 node6-2)
		(conn node7-2 node7-3)
		(conn node7-2 node7-1)
		(conn node7-3 node6-3)
		(conn node7-3 node7-4)
		(conn node7-3 node7-2)
		(conn node7-4 node6-4)
		(conn node7-4 node7-5)
		(conn node7-4 node7-3)
		(conn node7-5 node6-5)
		(conn node7-5 node7-6)
		(conn node7-5 node7-4)
		(conn node7-6 node6-6)
		(conn node7-6 node7-7)
		(conn node7-6 node7-5)
		(conn node7-7 node6-7)
		(conn node7-7 node7-6)
		(lock-shape node7-2 square)
		(lock-shape node7-3 square)
		(lock-shape node6-3 square)
		(lock-shape node5-3 square)
		(lock-shape node5-2 square)
		(lock-shape node5-1 square)
		(lock-shape node6-1 square)
		(lock-shape node7-1 square)
		(open node0-0)
		(open node0-1)
		(open node0-2)
		(open node0-3)
		(open node0-4)
		(open node0-5)
		(open node0-6)
		(open node0-7)
		(open node1-0)
		(open node1-1)
		(open node1-2)
		(open node1-3)
		(open node1-4)
		(open node1-5)
		(open node1-6)
		(open node1-7)
		(open node2-0)
		(open node2-1)
		(open node2-2)
		(open node2-3)
		(open node2-4)
		(open node2-5)
		(open node2-6)
		(open node2-7)
		(open node3-0)
		(open node3-1)
		(open node3-2)
		(open node3-3)
		(open node3-4)
		(open node3-5)
		(open node3-6)
		(open node3-7)
		(open node4-0)
		(open node4-1)
		(open node4-2)
		(open node4-3)
		(open node4-4)
		(open node4-5)
		(open node4-6)
		(open node4-7)
		(open node5-0)
		(open node5-4)
		(open node5-5)
		(open node5-6)
		(open node5-7)
		(open node6-0)
		(open node6-2)
		(open node6-4)
		(open node6-5)
		(open node6-6)
		(open node6-7)
		(open node7-0)
		(open node7-4)
		(open node7-5)
		(open node7-6)
		(open node7-7)
		(key key0)
		(key-shape key0 circle)
		(key key1)
		(key-shape key1 diamond)
		(key key2)
		(key-shape key2 triangle)
		(key key3)
		(key-shape key3 square)
		(key key4)
		(key-shape key4 diamond)
		(key key5)
		(key-shape key5 diamond)
		(key key6)
		(key-shape key6 square)
		(key key7)
		(key-shape key7 triangle)
		(key key8)
		(key-shape key8 square)
		(key key9)
		(key-shape key9 square)
		(key key10)
		(key-shape key10 square)
		(key key11)
		(key-shape key11 square)
		)
   (:goal (and (at key0 node1-1)
               (at key4 node6-4)
               (at key3 node3-0))))
