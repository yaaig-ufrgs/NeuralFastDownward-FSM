(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear b)
		(clear c)
		(clear e)
		(clear g)
		(clear k)
		(handempty)
		(on c a)
		(on d j)
		(on g l)
		(on h d)
		(on i f)
		(on j m)
		(on k o)
		(on l h)
		(on n i)
		(on o n)
		(ontable a)
		(ontable b)
		(ontable e)
		(ontable f)
		(ontable m)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)