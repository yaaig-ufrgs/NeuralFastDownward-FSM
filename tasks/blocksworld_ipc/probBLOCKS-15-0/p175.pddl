(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear b)
		(clear d)
		(clear j)
		(clear k)
		(clear l)
		(handempty)
		(on a o)
		(on c g)
		(on d f)
		(on e n)
		(on f e)
		(on h m)
		(on l c)
		(on m a)
		(on n h)
		(on o i)
		(ontable b)
		(ontable g)
		(ontable i)
		(ontable j)
		(ontable k)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
