(define (problem blocks-7-0)
(:domain blocks)
(:objects c f a b g d e )
(:init
	(clear d)
	(clear f)
	(clear g)
	(handempty)
	(on c e)
	(on d b)
	(on e a)
	(on g c)
	(ontable a)
	(ontable b)
	(ontable f)
)
(:goal (and (on a g) (on g d) (on d b) (on b c) (on c f) (on f e)))
)
