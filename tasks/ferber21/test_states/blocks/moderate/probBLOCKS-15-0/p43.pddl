(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear c)
		(clear e)
		(clear f)
		(clear j)
		(handempty)
		(on a o)
		(on b d)
		(on c g)
		(on d l)
		(on e k)
		(on h n)
		(on i a)
		(on j b)
		(on k h)
		(on l m)
		(on n i)
		(ontable f)
		(ontable g)
		(ontable m)
		(ontable o)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)