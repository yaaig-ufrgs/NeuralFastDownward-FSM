(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear f)
		(clear g)
		(clear k)
		(clear n)
		(handempty)
		(on a o)
		(on b a)
		(on c l)
		(on d h)
		(on e m)
		(on f e)
		(on g c)
		(on i b)
		(on j i)
		(on l d)
		(on m j)
		(ontable h)
		(ontable k)
		(ontable n)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
