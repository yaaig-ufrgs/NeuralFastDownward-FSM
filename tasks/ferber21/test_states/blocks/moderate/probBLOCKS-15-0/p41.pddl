(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear d)
		(clear g)
		(clear h)
		(clear k)
		(handempty)
		(on a o)
		(on b e)
		(on c l)
		(on e i)
		(on f b)
		(on h c)
		(on i a)
		(on j f)
		(on l n)
		(on m j)
		(on n m)
		(ontable d)
		(ontable g)
		(ontable k)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)