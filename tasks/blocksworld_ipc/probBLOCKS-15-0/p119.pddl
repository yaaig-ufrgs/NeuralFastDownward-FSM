(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear a)
		(clear e)
		(clear g)
		(handempty)
		(on a m)
		(on b o)
		(on c l)
		(on f b)
		(on g h)
		(on h c)
		(on i j)
		(on j k)
		(on k n)
		(on l d)
		(on m f)
		(on o i)
		(ontable d)
		(ontable e)
		(ontable n)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
