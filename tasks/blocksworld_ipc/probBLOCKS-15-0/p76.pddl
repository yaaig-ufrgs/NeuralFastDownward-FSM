(define (problem blocks-15-0)
(:domain blocks)
(:objects a c l d j h k o n g i f b m e )
		(:init
		(clear a)
		(clear b)
		(clear c)
		(clear g)
		(clear k)
		(clear l)
		(clear n)
		(handempty)
		(on a h)
		(on b o)
		(on d j)
		(on j m)
		(on k f)
		(on l e)
		(on n d)
		(on o i)
		(ontable c)
		(ontable e)
		(ontable f)
		(ontable g)
		(ontable h)
		(ontable i)
		(ontable m)
		)
(:goal (and (on g o) (on o h) (on h k) (on k m) (on m f) (on f e) (on e a)
            (on a b) (on b l) (on l j) (on j d) (on d n) (on n i) (on i c)))
)
