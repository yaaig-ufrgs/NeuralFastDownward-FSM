(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear c)
		(clear g)
		(clear h)
		(clear j)
		(handempty)
		(on a o)
		(on b k)
		(on c f)
		(on d m)
		(on g b)
		(on h i)
		(on i a)
		(on j l)
		(on l n)
		(on m e)
		(on n d)
		(ontable e)
		(ontable f)
		(ontable k)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
