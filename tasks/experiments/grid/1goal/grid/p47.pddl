(define (problem strips-grid-y-1)
   (:domain grid)
   (:objects node0-0 node0-1 node0-2 node0-3
             node1-0 node1-1 node1-2 node1-3
             node2-0 node2-1 node2-2 node2-3
             node3-0 node3-1 node3-2 node3-3
             square circle
             key0 key1 key2)
   (:init
	(arm-empty)
	(at key0 node2-3)
	(at key1 node1-3)
	(at key2 node3-0)
	(at-robot node1-0)
	(locked node2-2)
	(locked node2-3)
	(locked node3-2)
	(locked node3-3)
	(place node0-0)
	(place node0-1)
	(place node0-2)
	(place node0-3)
	(place node1-0)
	(place node1-1)
	(place node1-2)
	(place node1-3)
	(place node2-0)
	(place node2-1)
	(place node2-2)
	(place node2-3)
	(place node3-0)
	(place node3-1)
	(place node3-2)
	(place node3-3)
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
	(conn node0-3 node0-2)
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
	(conn node1-3 node1-2)
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
	(conn node2-3 node2-2)
	(conn node3-0 node2-0)
	(conn node3-0 node3-1)
	(conn node3-1 node2-1)
	(conn node3-1 node3-2)
	(conn node3-1 node3-0)
	(conn node3-2 node2-2)
	(conn node3-2 node3-3)
	(conn node3-2 node3-1)
	(conn node3-3 node2-3)
	(conn node3-3 node3-2)
	(lock-shape node3-3 square)
	(lock-shape node2-3 square)
	(lock-shape node2-2 square)
	(lock-shape node3-2 square)
	(open node0-0)
	(open node0-1)
	(open node0-2)
	(open node0-3)
	(open node1-0)
	(open node1-1)
	(open node1-2)
	(open node1-3)
	(open node2-0)
	(open node2-1)
	(open node3-0)
	(open node3-1)
	(key key0)
	(key-shape key0 circle)
	(key key1)
	(key-shape key1 square)
	(key key2)
	(key-shape key2 circle)
)
   (:goal (and (at key0 node1-1))))

