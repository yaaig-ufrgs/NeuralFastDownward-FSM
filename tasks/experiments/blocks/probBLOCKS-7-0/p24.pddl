(define (problem blocks-7-0)
(:domain blocks)
(:objects c f a b g d e )
(:init
	(clear a)
	(clear d)
	(clear f)
	(clear g)
	(handempty)
	(on a c)
	(on c b)
	(on g e)
	(ontable b)
	(ontable d)
	(ontable e)
	(ontable f)
)
(:goal (and (on a g) (on g d) (on d b) (on b c) (on c f) (on f e)))
)
