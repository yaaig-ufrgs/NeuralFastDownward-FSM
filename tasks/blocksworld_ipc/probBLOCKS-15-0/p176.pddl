(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear e)
		(clear f)
		(clear i)
		(clear m)
		(handempty)
		(on b c)
		(on c a)
		(on d h)
		(on e o)
		(on f d)
		(on g n)
		(on h j)
		(on j k)
		(on l g)
		(on m l)
		(on n b)
		(ontable a)
		(ontable i)
		(ontable k)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
